use crate::types::*;
use std::env;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

/// Classify an HTTP error status + body into an LlmError variant.
pub fn classify_error(status: u16, body: &str) -> LlmError {
    match status {
        429 | 500 | 502 | 503 => LlmError::Transient {
            status,
            message: body.to_string(),
        },
        401 | 403 => LlmError::Auth {
            status,
            message: body.to_string(),
        },
        _ => {
            let lower = body.to_lowercase();
            if lower.contains("context") || lower.contains("token") || lower.contains("too long") {
                LlmError::Overflow {
                    message: body.to_string(),
                }
            } else {
                LlmError::Fatal {
                    message: format!("HTTP {}: {}", status, body),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Retry policy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub jitter_factor: f64,
}

impl RetryPolicy {
    pub fn from_env() -> Self {
        Self {
            max_attempts: env::var("RETRY_MAX_ATTEMPTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3),
            base_delay_ms: env::var("RETRY_BASE_DELAY_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000),
            max_delay_ms: 30_000,
            jitter_factor: 0.25,
        }
    }
}

// ---------------------------------------------------------------------------
// API key rotation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ApiKeyEntry {
    key: String,
    cooldown_until: Option<std::time::Instant>,
}

#[derive(Debug, Clone)]
pub struct AuthProfile {
    keys: Vec<ApiKeyEntry>,
    current_index: usize,
}

impl AuthProfile {
    /// Load keys from environment: ANTHROPIC_API_KEY, ANTHROPIC_API_KEY_2, ...
    pub fn from_env(prefix: &str) -> Self {
        let mut keys = Vec::new();

        // Primary key
        if let Ok(key) = env::var(prefix) {
            keys.push(ApiKeyEntry {
                key,
                cooldown_until: None,
            });
        }

        // Additional keys: {PREFIX}_2, {PREFIX}_3, ...
        for i in 2..=10 {
            if let Ok(key) = env::var(format!("{}_{}", prefix, i)) {
                keys.push(ApiKeyEntry {
                    key,
                    cooldown_until: None,
                });
            } else {
                break;
            }
        }

        Self {
            keys,
            current_index: 0,
        }
    }

    /// Empty profile (no rotation).
    pub fn empty() -> Self {
        Self {
            keys: Vec::new(),
            current_index: 0,
        }
    }

    /// Try to rotate to the next available key. Returns true if rotated.
    fn rotate(&mut self) -> bool {
        if self.keys.len() <= 1 {
            return false;
        }
        let now = std::time::Instant::now();
        // Mark current key on cooldown for 60s
        self.keys[self.current_index].cooldown_until =
            Some(now + std::time::Duration::from_secs(60));

        // Find next key not on cooldown
        for offset in 1..self.keys.len() {
            let idx = (self.current_index + offset) % self.keys.len();
            if self.keys[idx]
                .cooldown_until
                .is_none_or(|until| now >= until)
            {
                self.current_index = idx;
                return true;
            }
        }
        false
    }

    /// Get the current API key, if any.
    /// NOTE: Key rotation updates this index, but the inner Llm backend
    /// reads its key at construction time. Full rotation requires backends
    /// to accept a key override per-request (future work).
    pub fn current_key(&self) -> Option<&str> {
        self.keys.get(self.current_index).map(|e| e.key.as_str())
    }
}

// ---------------------------------------------------------------------------
// ResilientLlm wrapper
// ---------------------------------------------------------------------------

pub struct ResilientLlm {
    inner: Box<dyn Llm>,
    policy: RetryPolicy,
    auth: AuthProfile,
}

impl ResilientLlm {
    pub fn new(inner: Box<dyn Llm>, policy: RetryPolicy, auth: AuthProfile) -> Self {
        Self {
            inner,
            policy,
            auth,
        }
    }

    fn compute_delay(&self, attempt: usize) -> std::time::Duration {
        let base = self.policy.base_delay_ms as f64;
        let exp = base * (2.0_f64).powi(attempt as i32);
        let capped = exp.min(self.policy.max_delay_ms as f64);
        // Simple deterministic jitter: reduce by jitter_factor * attempt fraction
        let jitter = capped
            * self.policy.jitter_factor
            * ((attempt as f64 + 1.0) / self.policy.max_attempts as f64);
        let delay_ms = (capped + jitter) as u64;
        std::time::Duration::from_millis(delay_ms)
    }
}

impl Llm for ResilientLlm {
    fn create(&mut self, params: LlmParams) -> Result<LlmResponse, LlmError> {
        let mut last_error = None;

        for attempt in 0..self.policy.max_attempts {
            match self.inner.create(params.clone()) {
                Ok(response) => return Ok(response),
                Err(LlmError::Transient { status, message }) => {
                    eprintln!(
                        "[resilience] Transient error (HTTP {}) on attempt {}/{}. Retrying...",
                        status,
                        attempt + 1,
                        self.policy.max_attempts
                    );
                    last_error = Some(LlmError::Transient { status, message });
                    std::thread::sleep(self.compute_delay(attempt));
                }
                Err(LlmError::Auth { status, message }) => {
                    if self.auth.rotate() {
                        eprintln!(
                            "[resilience] Auth error (HTTP {}). Rotated to next API key.",
                            status
                        );
                        last_error = Some(LlmError::Auth { status, message });
                        continue;
                    }
                    return Err(LlmError::Auth { status, message });
                }
                Err(e @ LlmError::Overflow { .. }) => return Err(e),
                Err(e @ LlmError::Fatal { .. }) => return Err(e),
            }
        }

        Err(last_error.unwrap_or(LlmError::Fatal {
            message: "Max retry attempts exceeded".to_string(),
        }))
    }
}

// ---------------------------------------------------------------------------
// GAP 1: ContextGuard — 3-stage overflow recovery decorator
// ---------------------------------------------------------------------------
//
// Wrapping order: AnthropicLlm → ResilientLlm → ContextGuard
//
// Stage 1: Normal call → on success or non-Overflow, return
// Stage 2: On Overflow → truncate oversized tool results (>2000 chars) → retry
// Stage 3: Still Overflow → LLM-summarize history (50% reduction) → retry
// Stage 4: Still Overflow → propagate error

pub struct ContextGuard {
    inner: Box<dyn Llm>,
    transcript_dir: PathBuf,
}

impl ContextGuard {
    pub fn new(inner: Box<dyn Llm>, transcript_dir: &Path) -> Self {
        Self {
            inner,
            transcript_dir: transcript_dir.to_path_buf(),
        }
    }
}

impl Llm for ContextGuard {
    fn create(&mut self, params: LlmParams) -> Result<LlmResponse, LlmError> {
        // Stage 1: Normal call
        match self.inner.create(params.clone()) {
            Ok(resp) => return Ok(resp),
            Err(LlmError::Overflow { message }) => {
                eprintln!(
                    "[context-guard] Overflow (stage 1), truncating tool results: {}",
                    &message[..message.len().min(100)]
                );
            }
            Err(e) => return Err(e),
        }

        // Stage 2: Truncate oversized tool results and retry
        let mut params2 = params;
        let truncated = crate::memory::truncate_tool_results(&mut params2.messages, 2000);
        if truncated {
            match self.inner.create(params2.clone()) {
                Ok(resp) => return Ok(resp),
                Err(LlmError::Overflow { message }) => {
                    eprintln!(
                        "[context-guard] Overflow (stage 2), compacting history: {}",
                        &message[..message.len().min(100)]
                    );
                }
                Err(e) => return Err(e),
            }
        }

        // Stage 3: LLM-summarize history for ~50% reduction
        let compacted = crate::memory::compact_for_overflow(
            &params2.messages,
            self.inner.as_mut(),
            &self.transcript_dir,
        );
        let mut params3 = params2;
        params3.messages = compacted;

        // Final retry — propagate whatever happens
        self.inner.create(params3)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockLLM;
    use crate::types::ContentBlock;

    fn quick_policy() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            base_delay_ms: 1, // 1ms for tests
            max_delay_ms: 10,
            jitter_factor: 0.0,
        }
    }

    fn make_params() -> LlmParams {
        LlmParams {
            model: "test".to_string(),
            system: "sys".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
        }
    }

    #[test]
    fn resilient_success_path() {
        let mut mock = MockLLM::new();
        mock.queue(
            "end_turn",
            vec![ContentBlock::Text {
                text: "hello".into(),
            }],
        );

        let mut llm = ResilientLlm::new(Box::new(mock), quick_policy(), AuthProfile::empty());
        let resp = llm.create(make_params()).unwrap();
        assert_eq!(resp.stop_reason, "end_turn");
    }

    #[test]
    fn resilient_transient_then_success() {
        let mut mock = MockLLM::new();
        mock.queue_error(LlmError::Transient {
            status: 429,
            message: "rate limited".into(),
        });
        mock.queue("end_turn", vec![ContentBlock::Text { text: "ok".into() }]);

        let mut llm = ResilientLlm::new(Box::new(mock), quick_policy(), AuthProfile::empty());
        let resp = llm.create(make_params()).unwrap();
        assert_eq!(resp.stop_reason, "end_turn");
    }

    #[test]
    fn resilient_max_retries_exhausted() {
        let mut mock = MockLLM::new();
        for _ in 0..3 {
            mock.queue_error(LlmError::Transient {
                status: 503,
                message: "unavailable".into(),
            });
        }

        let mut llm = ResilientLlm::new(Box::new(mock), quick_policy(), AuthProfile::empty());
        let err = llm.create(make_params()).unwrap_err();
        assert!(matches!(err, LlmError::Transient { .. }));
    }

    #[test]
    fn resilient_overflow_not_retried() {
        let mut mock = MockLLM::new();
        mock.queue_error(LlmError::Overflow {
            message: "too long".into(),
        });

        let mut llm = ResilientLlm::new(Box::new(mock), quick_policy(), AuthProfile::empty());
        let err = llm.create(make_params()).unwrap_err();
        assert!(matches!(err, LlmError::Overflow { .. }));
    }

    #[test]
    fn resilient_fatal_not_retried() {
        let mut mock = MockLLM::new();
        mock.queue_error(LlmError::Fatal {
            message: "bad request".into(),
        });

        let mut llm = ResilientLlm::new(Box::new(mock), quick_policy(), AuthProfile::empty());
        let err = llm.create(make_params()).unwrap_err();
        assert!(matches!(err, LlmError::Fatal { .. }));
    }

    #[test]
    fn context_guard_normal_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let mut mock = MockLLM::new();
        mock.queue(
            "end_turn",
            vec![ContentBlock::Text {
                text: "fine".into(),
            }],
        );

        let mut guard = ContextGuard::new(Box::new(mock), dir.path());
        let resp = guard.create(make_params()).unwrap();
        assert_eq!(resp.stop_reason, "end_turn");
    }

    #[test]
    fn context_guard_overflow_triggers_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let mut mock = MockLLM::new();
        // First call: overflow
        mock.queue_error(LlmError::Overflow {
            message: "context too long".into(),
        });
        // Stage 2 retry after truncation: success
        mock.queue(
            "end_turn",
            vec![ContentBlock::Text {
                text: "recovered".into(),
            }],
        );

        let mut guard = ContextGuard::new(Box::new(mock), dir.path());

        // Build params with a large tool_result to trigger truncation
        let mut params = make_params();
        params.messages = vec![
            crate::types::Message {
                role: "user".into(),
                content: crate::types::MessageContent::Text("hi".into()),
            },
            crate::types::Message {
                role: "assistant".into(),
                content: crate::types::MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({}),
                }]),
            },
            crate::types::Message {
                role: "user".into(),
                content: crate::types::MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".into(),
                    content: "x".repeat(5000), // oversized
                }]),
            },
        ];

        let resp = guard.create(params).unwrap();
        assert_eq!(resp.stop_reason, "end_turn");
    }
}

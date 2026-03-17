use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::channels::ChannelManager;
use crate::concurrency::BackgroundManager;
use crate::context::Projector;
use crate::delivery::DeliveryQueue;
use crate::heartbeat::HeartbeatManager;
use crate::isolation::EventBus;
use crate::knowledge::SkillLoader;
use crate::memory::SessionStore;
use crate::memory_store::MemoryStore;
use crate::planning::{NagPolicy, TodoManager};
use crate::protocols::RequestTracker;
use crate::tasks::TaskManager;
use crate::teams::{MessageBus, TeammateManager};
use crate::types::{CompactSignal, ToolEvent};

// ---------------------------------------------------------------------------
// HandlerError + HandlerResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum HandlerError {
    Validation { message: String },
    Execution { message: String },
    Timeout { elapsed_ms: u64 },
}

impl fmt::Display for HandlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HandlerError::Validation { message } => write!(f, "Validation error: {}", message),
            HandlerError::Execution { message } => write!(f, "Error: {}", message),
            HandlerError::Timeout { elapsed_ms } => {
                write!(f, "Timeout after {}ms", elapsed_ms)
            }
        }
    }
}

impl std::error::Error for HandlerError {}

pub type HandlerResult = Result<String, HandlerError>;

// ---------------------------------------------------------------------------
// JSON input extraction helpers
// ---------------------------------------------------------------------------

/// Extract a required string field from JSON input, or return a Validation error.
pub fn require_str<'a>(input: &'a serde_json::Value, field: &str) -> Result<&'a str, HandlerError> {
    input
        .get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| HandlerError::Validation {
            message: format!("missing '{}' field", field),
        })
}

/// Extract a required i64 field from JSON input, or return a Validation error.
pub fn require_i64(input: &serde_json::Value, field: &str) -> Result<i64, HandlerError> {
    input
        .get(field)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| HandlerError::Validation {
            message: format!("missing '{}' field", field),
        })
}

/// Convert any Display-able error into `HandlerError::Execution`.
pub fn exec_err(e: impl fmt::Display) -> HandlerError {
    HandlerError::Execution {
        message: format!("{}", e),
    }
}

// ---------------------------------------------------------------------------
// Handler trait
// ---------------------------------------------------------------------------

pub trait Handler: Send + Sync {
    fn call(&self, ctx: &AgentContext, input: serde_json::Value) -> HandlerResult;
}

// Blanket impl for closures
impl<F> Handler for F
where
    F: Fn(&AgentContext, serde_json::Value) -> HandlerResult + Send + Sync,
{
    fn call(&self, ctx: &AgentContext, input: serde_json::Value) -> HandlerResult {
        (self)(ctx, input)
    }
}

// ---------------------------------------------------------------------------
// Middle + Chain
// ---------------------------------------------------------------------------

pub type Middle = Arc<dyn Fn(Arc<dyn Handler>) -> Arc<dyn Handler> + Send + Sync>;

pub struct Chain {
    inner: Arc<dyn Handler>,
}

impl Chain {
    pub fn new(handler: Arc<dyn Handler>) -> Self {
        Self { inner: handler }
    }

    pub fn with(self, middleware: Middle) -> Self {
        Self {
            inner: middleware(self.inner),
        }
    }

    pub fn build(self) -> Arc<dyn Handler> {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// AgentContext sub-structs
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AgentIdentity {
    pub name: String,
    pub role: String,
    pub id: String,
    pub session_id: String,
}

#[derive(Clone)]
pub struct AgentServices {
    pub todo: Arc<RwLock<TodoManager>>,
    pub nag: Arc<Mutex<NagPolicy>>,
    pub skill_loader: Arc<SkillLoader>,
    pub task_manager: Arc<RwLock<TaskManager>>,
    pub bg: Arc<Mutex<BackgroundManager>>,
    pub message_bus: Arc<MessageBus>,
    pub teammate_manager: Arc<RwLock<TeammateManager>>,
    pub request_tracker: Arc<RequestTracker>,
    pub event_bus: Arc<EventBus>,
    pub heartbeat_manager: Arc<HeartbeatManager>,
    pub memory_store: Arc<MemoryStore>,
    pub delivery_queue: Arc<DeliveryQueue>,
    pub channels: Arc<Mutex<ChannelManager>>,
    pub sessions: Arc<SessionStore>,
}

#[derive(Clone)]
pub struct AgentSignals {
    pub compact: Arc<CompactSignal>,
    pub idle: Arc<AtomicBool>,
    pub interrupt: Option<Arc<AtomicBool>>,
}

// ---------------------------------------------------------------------------
// AgentContext
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AgentContext {
    pub identity: AgentIdentity,
    pub cwd: PathBuf,
    pub tasks_dir: PathBuf,
    pub llm_backend: String,
    pub services: AgentServices,
    pub signals: AgentSignals,
    pub projector: Option<Arc<Projector>>,
    pub transcript_dir: Option<PathBuf>,
    pub tool_callback: Option<Arc<dyn Fn(ToolEvent) + Send + Sync>>,
}

impl AgentContext {
    /// Create a child AgentContext for a spawned teammate or subagent.
    /// Inherits shared services but gets fresh identity and signals.
    pub fn child_context(
        &self,
        name: String,
        role: String,
        transcript_dir: Option<PathBuf>,
    ) -> Self {
        let mut ctx = self.clone();
        ctx.identity.name = name;
        ctx.identity.role = role;
        ctx.identity.id = uuid::Uuid::new_v4().to_string();
        ctx.identity.session_id = format!(
            "{}-child-{}",
            self.identity.session_id,
            &ctx.identity.id[..8]
        );
        ctx.signals = AgentSignals {
            compact: Arc::new(CompactSignal::new()),
            idle: Arc::new(AtomicBool::new(false)),
            interrupt: None,
        };
        ctx.tool_callback = None;
        ctx.projector = None;
        ctx.transcript_dir = transcript_dir;
        ctx
    }

    /// Create a minimal AgentContext for testing.
    pub fn mock(workspace: &Path) -> Self {
        let tasks_dir = workspace.join("tasks");
        std::fs::create_dir_all(&tasks_dir).ok();
        let events_path = workspace.join("events.jsonl");
        let inbox_dir = workspace.join("inbox");
        let team_dir = workspace.join("team");
        let skills_dir = workspace.join("skills");
        let memories_dir = workspace.join("memories");
        let memory_md = workspace.join("MEMORY.md");
        let delivery_dir = workspace.join("delivery");
        let session_dir = workspace.join("sessions");
        let cron_path = workspace.join("cron.json");

        Self {
            identity: AgentIdentity {
                name: "test-agent".to_string(),
                role: "tester".to_string(),
                id: "test-id".to_string(),
                session_id: "test-session".to_string(),
            },
            tasks_dir,
            llm_backend: "anthropic".to_string(),
            cwd: workspace.to_path_buf(),
            services: AgentServices {
                todo: Arc::new(RwLock::new(TodoManager::new())),
                nag: Arc::new(Mutex::new(NagPolicy::new(3))),
                skill_loader: Arc::new(SkillLoader::new(&skills_dir)),
                task_manager: Arc::new(RwLock::new(TaskManager::new(&workspace.join("tasks")))),
                bg: Arc::new(Mutex::new(BackgroundManager::new())),
                message_bus: Arc::new(MessageBus::new(&inbox_dir)),
                teammate_manager: Arc::new(RwLock::new(TeammateManager::new(&team_dir))),
                request_tracker: Arc::new(RequestTracker::new()),
                event_bus: Arc::new(EventBus::new(&events_path)),
                heartbeat_manager: Arc::new(HeartbeatManager::new(&cron_path)),
                memory_store: Arc::new(MemoryStore::new(&memory_md, &memories_dir)),
                delivery_queue: Arc::new(DeliveryQueue::new(&delivery_dir)),
                channels: Arc::new(Mutex::new(ChannelManager::new())),
                sessions: Arc::new(SessionStore::new(&session_dir, "test-session")),
            },
            signals: AgentSignals {
                compact: Arc::new(CompactSignal::new()),
                idle: Arc::new(AtomicBool::new(false)),
                interrupt: None,
            },
            projector: None,
            transcript_dir: None,
            tool_callback: None,
        }
    }
}

// ---------------------------------------------------------------------------
// HandlerRegistry
// ---------------------------------------------------------------------------

pub struct HandlerRegistry {
    handlers: HashMap<String, Arc<dyn Handler>>,
}

impl HandlerRegistry {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::with_capacity(32),
        }
    }

    pub fn register(&mut self, name: impl Into<String>, handler: Arc<dyn Handler>) {
        self.handlers.insert(name.into(), handler);
    }

    pub fn route(&self, ctx: &AgentContext, name: &str, input: serde_json::Value) -> HandlerResult {
        match self.handlers.get(name) {
            Some(handler) => handler.call(ctx, input),
            None => Err(HandlerError::Validation {
                message: format!("Unknown tool: {}", name),
            }),
        }
    }

    pub fn extend(&mut self, other: HandlerRegistry) {
        self.handlers.extend(other.handlers);
    }

    pub fn contains(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }
}

impl Default for HandlerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Middleware factories
// ---------------------------------------------------------------------------

/// Middleware that caps output size at `max_bytes`, truncating at a UTF-8 boundary.
pub fn with_output_cap(max_bytes: usize) -> Middle {
    Arc::new(move |inner: Arc<dyn Handler>| -> Arc<dyn Handler> {
        Arc::new(
            move |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let result = inner.call(ctx, input)?;
                if result.len() > max_bytes {
                    let mut truncated =
                        crate::util::truncate_at_boundary(&result, max_bytes).to_string();
                    truncated.push_str("\n... (output truncated)");
                    Ok(truncated)
                } else {
                    Ok(result)
                }
            },
        )
    })
}

/// Middleware that checks handler execution duration after completion.
/// NOTE: This does not preemptively cancel long-running handlers — it checks
/// elapsed time after the handler returns and returns a Timeout error if exceeded.
/// For true preemptive timeout, use BashHandler's built-in timeout parameter.
pub fn with_timeout(duration: Duration) -> Middle {
    Arc::new(move |inner: Arc<dyn Handler>| -> Arc<dyn Handler> {
        Arc::new(
            move |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let start = std::time::Instant::now();
                let result = inner.call(ctx, input);
                let elapsed = start.elapsed();
                if elapsed > duration {
                    return Err(HandlerError::Timeout {
                        elapsed_ms: elapsed.as_millis() as u64,
                    });
                }
                result
            },
        )
    })
}

/// Middleware that retries on `HandlerError::Execution` up to `max_attempts` times.
pub fn with_retry(max_attempts: usize) -> Middle {
    Arc::new(move |inner: Arc<dyn Handler>| -> Arc<dyn Handler> {
        Arc::new(
            move |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let mut last_err = None;
                for _ in 0..max_attempts {
                    match inner.call(ctx, input.clone()) {
                        Ok(result) => return Ok(result),
                        Err(HandlerError::Execution { message }) => {
                            last_err = Some(HandlerError::Execution { message });
                        }
                        Err(e) => return Err(e),
                    }
                }
                Err(last_err.unwrap())
            },
        )
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_handler_trait_closure_blanket_impl() {
        let handler: Arc<dyn Handler> = Arc::new(
            |_ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                Ok(format!("got: {}", input))
            },
        );
        let tmp = tempfile::tempdir().unwrap();
        let ctx = AgentContext::mock(tmp.path());
        let result = handler.call(&ctx, serde_json::json!({"key": "val"}));
        assert!(result.is_ok());
        assert!(result.unwrap().contains("key"));
    }

    #[test]
    fn test_handler_registry_route_and_contains() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = AgentContext::mock(tmp.path());
        let mut reg = HandlerRegistry::new();
        reg.register(
            "echo",
            Arc::new(
                |_ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                    Ok(input.to_string())
                },
            ),
        );
        assert!(reg.contains("echo"));
        assert!(!reg.contains("missing"));

        let result = reg.route(&ctx, "echo", serde_json::json!({"x": 1}));
        assert!(result.is_ok());

        let err = reg.route(&ctx, "missing", serde_json::json!({}));
        assert!(err.is_err());
    }

    #[test]
    fn test_handler_registry_extend() {
        let mut a = HandlerRegistry::new();
        a.register(
            "tool_a",
            Arc::new(|_: &AgentContext, _: serde_json::Value| -> HandlerResult { Ok("a".into()) }),
        );
        let mut b = HandlerRegistry::new();
        b.register(
            "tool_b",
            Arc::new(|_: &AgentContext, _: serde_json::Value| -> HandlerResult { Ok("b".into()) }),
        );
        a.extend(b);
        assert!(a.contains("tool_a"));
        assert!(a.contains("tool_b"));
    }

    #[test]
    fn test_chain_with_middleware() {
        let inner: Arc<dyn Handler> = Arc::new(
            |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok("x".repeat(200))
            },
        );
        let chained = Chain::new(inner).with(with_output_cap(50)).build();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = AgentContext::mock(tmp.path());
        let result = chained.call(&ctx, serde_json::json!({})).unwrap();
        assert!(result.len() < 200);
        assert!(result.contains("truncated"));
    }

    #[test]
    fn test_with_output_cap_passthrough_small() {
        let inner: Arc<dyn Handler> = Arc::new(
            |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok("small".into())
            },
        );
        let chained = Chain::new(inner).with(with_output_cap(1000)).build();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = AgentContext::mock(tmp.path());
        let result = chained.call(&ctx, serde_json::json!({})).unwrap();
        assert_eq!(result, "small");
    }

    #[test]
    fn test_with_retry_succeeds_on_second_attempt() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let inner: Arc<dyn Handler> = Arc::new(
            move |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                let n = counter_clone.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(HandlerError::Execution {
                        message: "transient".into(),
                    })
                } else {
                    Ok("ok".into())
                }
            },
        );
        let chained = Chain::new(inner).with(with_retry(3)).build();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = AgentContext::mock(tmp.path());
        let result = chained.call(&ctx, serde_json::json!({}));
        assert_eq!(result, Ok("ok".into()));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_with_retry_does_not_retry_validation_errors() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let inner: Arc<dyn Handler> = Arc::new(
            move |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Err(HandlerError::Validation {
                    message: "bad input".into(),
                })
            },
        );
        let chained = Chain::new(inner).with(with_retry(3)).build();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = AgentContext::mock(tmp.path());
        let result = chained.call(&ctx, serde_json::json!({}));
        assert!(result.is_err());
        // Should not retry validation errors — only called once
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_handler_error_display() {
        let v = HandlerError::Validation {
            message: "bad".into(),
        };
        assert!(format!("{}", v).contains("bad"));

        let e = HandlerError::Execution {
            message: "fail".into(),
        };
        assert!(format!("{}", e).contains("fail"));

        let t = HandlerError::Timeout { elapsed_ms: 5000 };
        assert!(format!("{}", t).contains("5000"));
    }

    #[test]
    fn test_agent_context_mock_clone() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = AgentContext::mock(tmp.path());
        let cloned = ctx.clone();
        assert_eq!(ctx.identity.name, cloned.identity.name);
        assert_eq!(ctx.cwd, cloned.cwd);
    }
}

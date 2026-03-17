use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

// ---------------------------------------------------------------------------
// Message content: either a plain string or a vec of blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Blocks(Vec<ContentBlock>),
    Text(String),
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

// ---------------------------------------------------------------------------
// LLM abstraction
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LlmParams {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<serde_json::Value>,
    pub max_tokens: usize,
}

#[derive(Debug)]
pub struct LlmResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: String,
}

// ---------------------------------------------------------------------------
// LLM errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum LlmError {
    /// Retryable: 429, 500, 502, 503
    Transient { status: u16, message: String },
    /// Context window exceeded
    Overflow { message: String },
    /// Authentication failure: 401, 403
    Auth { status: u16, message: String },
    /// Non-retryable
    Fatal { message: String },
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Transient { status, message } => {
                write!(f, "Transient error (HTTP {}): {}", status, message)
            }
            LlmError::Overflow { message } => write!(f, "Context overflow: {}", message),
            LlmError::Auth { status, message } => {
                write!(f, "Auth error (HTTP {}): {}", status, message)
            }
            LlmError::Fatal { message } => write!(f, "Fatal error: {}", message),
        }
    }
}

impl std::error::Error for LlmError {}

pub trait Llm {
    fn create(&mut self, params: LlmParams) -> Result<LlmResponse, LlmError>;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum AgentError {
    ValueError(String),
    NotFound(String),
    Io(std::io::Error),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::ValueError(msg) => write!(f, "ValueError: {}", msg),
            AgentError::NotFound(msg) => write!(f, "NotFound: {}", msg),
            AgentError::Io(err) => write!(f, "IO: {}", err),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<std::io::Error> for AgentError {
    fn from(err: std::io::Error) -> Self {
        AgentError::Io(err)
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

pub type ToolHandler = Box<dyn Fn(serde_json::Value) -> String + Send + Sync>;
pub type Dispatch = HashMap<String, ToolHandler>;

// ---------------------------------------------------------------------------
// CompactSignal: allows LLM to request on-demand compaction
// ---------------------------------------------------------------------------

pub struct CompactSignal {
    requested: AtomicBool,
}

impl CompactSignal {
    pub fn new() -> Self {
        Self {
            requested: AtomicBool::new(false),
        }
    }

    pub fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }

    /// Returns true if compaction was requested, and clears the flag.
    pub fn take(&self) -> bool {
        self.requested.swap(false, Ordering::Acquire)
    }
}

impl Default for CompactSignal {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ToolEvent: emitted during tool execution for UI feedback
// ---------------------------------------------------------------------------

pub enum ToolEvent {
    Start {
        name: String,
        input: serde_json::Value,
    },
    Complete {
        name: String,
        summary: String,
        duration: std::time::Duration,
    },
    Error {
        name: String,
        error: String,
        duration: std::time::Duration,
    },
}

// ---------------------------------------------------------------------------
// LoopSignals: bundles optional signals for run_agent_loop
// ---------------------------------------------------------------------------

pub struct LoopSignals<'a> {
    pub compact_signal: Option<&'a CompactSignal>,
    pub transcript_dir: Option<&'a std::path::Path>,
    pub idle_signal: Option<&'a AtomicBool>,
    pub tool_callback: Option<&'a dyn Fn(ToolEvent)>,
    pub interrupt_signal: Option<&'a AtomicBool>,
    pub projector: Option<&'a crate::context::Projector>,
}

impl<'a> LoopSignals<'a> {
    pub fn none() -> Self {
        Self {
            compact_signal: None,
            transcript_dir: None,
            idle_signal: None,
            tool_callback: None,
            interrupt_signal: None,
            projector: None,
        }
    }
}

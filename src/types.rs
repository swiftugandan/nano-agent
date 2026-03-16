use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
    },
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

pub struct LlmParams {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<serde_json::Value>,
    pub max_tokens: usize,
}

pub struct LlmResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: String,
}

pub trait Llm {
    fn create(&mut self, params: LlmParams) -> LlmResponse;
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

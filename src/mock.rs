use crate::handler::{AgentContext, HandlerRegistry, HandlerResult};
use crate::types::*;
use std::collections::VecDeque;
use std::sync::Arc;

/// Mock LLM that returns pre-queued responses. Mirrors Python conftest.py MockLLM.
pub struct MockLLM {
    responses: VecDeque<Result<LlmResponse, LlmError>>,
    calls: Vec<LlmParams>,
}

impl Default for MockLLM {
    fn default() -> Self {
        Self::new()
    }
}

impl MockLLM {
    pub fn new() -> Self {
        Self {
            responses: VecDeque::new(),
            calls: Vec::new(),
        }
    }

    /// Queue a successful response with the given stop_reason and content blocks.
    pub fn queue(&mut self, stop_reason: &str, content: Vec<ContentBlock>) {
        self.responses.push_back(Ok(LlmResponse {
            content,
            stop_reason: stop_reason.to_string(),
        }));
    }

    /// Queue an error response.
    pub fn queue_error(&mut self, error: LlmError) {
        self.responses.push_back(Err(error));
    }

    /// Queue the same response `count` times, replacing `{i}` in tool_use ids.
    pub fn queue_repeat(&mut self, stop_reason: &str, content: Vec<ContentBlock>, count: usize) {
        for i in 0..count {
            let adjusted: Vec<ContentBlock> = content
                .iter()
                .map(|block| match block {
                    ContentBlock::ToolUse { id, name, input } => ContentBlock::ToolUse {
                        id: id.replace("{i}", &i.to_string()),
                        name: name.clone(),
                        input: input.clone(),
                    },
                    other => other.clone(),
                })
                .collect();
            self.queue(stop_reason, adjusted);
        }
    }

    pub fn call_count(&self) -> usize {
        self.calls.len()
    }

    pub fn calls(&self) -> &[LlmParams] {
        &self.calls
    }
}

impl Llm for MockLLM {
    fn create(&mut self, params: LlmParams) -> Result<LlmResponse, LlmError> {
        self.calls.push(params);
        if let Some(result) = self.responses.pop_front() {
            result
        } else {
            Ok(LlmResponse {
                content: vec![ContentBlock::Text {
                    text: "(no more responses)".to_string(),
                }],
                stop_reason: "end_turn".to_string(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Create a handler registry from {name: return_value} pairs.
pub fn make_registry(handlers: std::collections::HashMap<String, String>) -> HandlerRegistry {
    let mut reg = HandlerRegistry::new();
    for (name, val) in handlers {
        reg.register(
            name,
            Arc::new(
                move |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                    Ok(val.clone())
                },
            ),
        );
    }
    reg
}

/// Create a text content block.
pub fn make_text_block(text: &str) -> ContentBlock {
    ContentBlock::Text {
        text: text.to_string(),
    }
}

/// Create a tool_use content block.
pub fn make_tool_use_block(id: &str, name: &str, input: serde_json::Value) -> ContentBlock {
    ContentBlock::ToolUse {
        id: id.to_string(),
        name: name.to_string(),
        input,
    }
}

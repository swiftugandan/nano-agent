use crate::handler::{AgentContext, HandlerRegistry};
use crate::tools::tool_definitions;
use crate::types::*;
use std::sync::OnceLock;

static CHILD_TOOLS: OnceLock<Vec<serde_json::Value>> = OnceLock::new();
static CHILD_TOOL_NAMES: OnceLock<Vec<String>> = OnceLock::new();

/// Get child tools derived from tool_definitions (parsed once).
pub fn child_tools() -> &'static Vec<serde_json::Value> {
    CHILD_TOOLS.get_or_init(tool_definitions)
}

/// Get child tool names (computed once).
pub fn child_tool_names() -> &'static Vec<String> {
    CHILD_TOOL_NAMES.get_or_init(|| {
        child_tools()
            .iter()
            .filter_map(|t| t["name"].as_str().map(|s| s.to_string()))
            .collect()
    })
}

/// Subagent factory: spawn a child agent with fresh messages.
pub struct SubagentFactory;

impl SubagentFactory {
    /// Spawn a subagent with fresh context. Returns only the final text.
    /// Creates a child AgentContext derived from the parent (new agent_id, shared services).
    pub fn spawn(
        llm: &mut dyn Llm,
        prompt: &str,
        tools: &[serde_json::Value],
        registry: &HandlerRegistry,
        ctx: &AgentContext,
        max_iterations: usize,
    ) -> String {
        // Create isolated child context with new agent_id and fresh signals
        let child_ctx = ctx.child_context(
            format!("subagent-{}", &uuid::Uuid::new_v4().to_string()[..8]),
            "subagent".to_string(),
            None,
        );

        let mut sub_messages: Vec<serde_json::Value> = vec![serde_json::json!({
            "role": "user",
            "content": prompt,
        })];

        let mut last_response_content: Vec<ContentBlock> = Vec::new();

        for _ in 0..max_iterations {
            let typed_messages: Vec<Message> = sub_messages
                .iter()
                .filter_map(|m| serde_json::from_value(m.clone()).ok())
                .collect();

            let response = match llm.create(LlmParams {
                model: "default".to_string(),
                system: "You are a focused sub-agent. Complete the given task using the available tools.".to_string(),
                messages: typed_messages,
                tools: tools.to_vec(),
                max_tokens: 16_000,
            }) {
                Ok(r) => r,
                Err(e) => {
                    return format!("(subagent error: {})", e);
                }
            };

            let content_json: Vec<serde_json::Value> = response
                .content
                .iter()
                .map(|b| serde_json::to_value(b).unwrap())
                .collect();
            sub_messages.push(serde_json::json!({
                "role": "assistant",
                "content": content_json,
            }));

            last_response_content = response.content.clone();

            if response.stop_reason != "tool_use" {
                break;
            }

            let mut results: Vec<serde_json::Value> = Vec::new();
            for block in &response.content {
                if let ContentBlock::ToolUse { id, name, input } = block {
                    let output = match registry.route(&child_ctx, name, input.clone()) {
                        Ok(s) => s,
                        Err(e) => format!("Error: {}", e),
                    };
                    results.push(serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": output,
                    }));
                }
            }
            sub_messages.push(serde_json::json!({
                "role": "user",
                "content": results,
            }));
        }

        // Extract only final text
        let text: String = last_response_content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            "(no summary)".to_string()
        } else {
            text
        }
    }
}

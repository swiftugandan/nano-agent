use crate::types::*;
use std::path::{Path, PathBuf};

/// Run the core agent loop pattern with an LLM.
/// Returns the number of LLM calls made.
pub fn run_agent_loop(
    llm: &mut dyn Llm,
    system: &str,
    messages: &mut Vec<serde_json::Value>,
    tools: &[serde_json::Value],
    dispatch: &Dispatch,
) -> usize {
    let mut call_count = 0;
    loop {
        let typed_messages: Vec<Message> = messages
            .iter()
            .filter_map(|m| serde_json::from_value(m.clone()).ok())
            .collect();

        let response = llm.create(LlmParams {
            model: "default".to_string(),
            system: system.to_string(),
            messages: typed_messages,
            tools: tools.to_vec(),
            max_tokens: 16_000,
        });
        call_count += 1;

        // Serialize content blocks for the assistant message
        let content_json: Vec<serde_json::Value> = response
            .content
            .iter()
            .map(|block| serde_json::to_value(block).unwrap())
            .collect();

        messages.push(serde_json::json!({
            "role": "assistant",
            "content": content_json,
        }));

        if response.stop_reason != "tool_use" {
            break;
        }

        // Execute tools and collect results
        let mut results: Vec<serde_json::Value> = Vec::new();
        for block in &response.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                let output = route(dispatch, name, input.clone());
                results.push(serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": output,
                }));
            }
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": results,
        }));
    }
    call_count
}

/// Route a tool call to the correct handler. Never panics.
/// Returns "Unknown tool: {name}" for missing handlers.
pub fn route(dispatch: &Dispatch, tool_name: &str, input: serde_json::Value) -> String {
    match dispatch.get(tool_name) {
        Some(handler) => handler(input),
        None => format!("Unknown tool: {}", tool_name),
    }
}

/// Path sandbox: ensures paths stay within the workspace.
pub struct PathSandbox {
    workspace: PathBuf,
}

impl PathSandbox {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
        }
    }

    /// Resolve a relative path within the workspace. Rejects escape attempts.
    pub fn safe_path(&self, relative: &str) -> Result<PathBuf, AgentError> {
        let joined = self.workspace.join(relative);
        // Normalize the path by resolving . and .. components manually
        let mut components = Vec::new();
        for component in joined.components() {
            match component {
                std::path::Component::ParentDir => {
                    components.pop();
                }
                std::path::Component::CurDir => {}
                other => components.push(other),
            }
        }
        let resolved: PathBuf = components.iter().collect();

        if !resolved.starts_with(&self.workspace) {
            return Err(AgentError::ValueError(format!(
                "Path escapes workspace: {}",
                relative
            )));
        }
        Ok(resolved)
    }
}

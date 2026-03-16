use crate::types::*;

/// Child tools: bash only, no recursive spawn (no "task" or "subagent").
pub const CHILD_TOOLS_JSON: &str = r#"[
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}
]"#;

/// Get CHILD_TOOLS as parsed JSON values.
pub fn child_tools() -> Vec<serde_json::Value> {
    serde_json::from_str(CHILD_TOOLS_JSON).unwrap()
}

/// Get child tool names.
pub fn child_tool_names() -> Vec<String> {
    child_tools()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect()
}

/// Subagent factory: spawn a child agent with fresh messages.
pub struct SubagentFactory;

impl SubagentFactory {
    /// Spawn a subagent with fresh context. Returns only the final text.
    pub fn spawn(
        llm: &mut dyn Llm,
        prompt: &str,
        tools: &[serde_json::Value],
        dispatch: &Dispatch,
        max_iterations: usize,
    ) -> String {
        let mut sub_messages: Vec<serde_json::Value> = vec![serde_json::json!({
            "role": "user",
            "content": prompt,
        })];

        let mut last_response_content: Vec<ContentBlock> = Vec::new();

        for _ in 0..max_iterations {
            let response = llm.create(LlmParams {
                model: "test".to_string(),
                system: "Test".to_string(),
                messages: Vec::new(),
                tools: tools.to_vec(),
                max_tokens: 8000,
            });

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
                    let handler = dispatch.get(name.as_str());
                    let output = match handler {
                        Some(h) => h(input.clone()),
                        None => format!("Unknown tool: {}", name),
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

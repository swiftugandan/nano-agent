use crate::types::*;
use std::path::{Path, PathBuf};

pub const KEEP_RECENT: usize = 3;
pub const THRESHOLD: usize = 50_000;

/// Rough token count: ~4 chars per token.
pub fn estimate_tokens(messages: &[serde_json::Value]) -> usize {
    let s = serde_json::to_string(messages).unwrap_or_default();
    s.len() / 4
}

/// Layer 1: micro_compact — replace old tool_results with placeholders.
/// Preserves the last KEEP_RECENT tool_results.
pub fn micro_compact(messages: &mut Vec<serde_json::Value>) {
    // Collect indices of all tool_result entries
    struct ToolResultRef {
        msg_idx: usize,
        part_idx: usize,
    }

    let mut tool_results: Vec<ToolResultRef> = Vec::new();

    for (msg_idx, msg) in messages.iter().enumerate() {
        if msg["role"].as_str() == Some("user") {
            if let Some(content) = msg["content"].as_array() {
                for (part_idx, part) in content.iter().enumerate() {
                    if part["type"].as_str() == Some("tool_result") {
                        tool_results.push(ToolResultRef { msg_idx, part_idx });
                    }
                }
            }
        }
    }

    if tool_results.len() <= KEEP_RECENT {
        return;
    }

    // Build a map of tool_use_id -> tool_name from assistant messages
    let mut tool_name_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for msg in messages.iter() {
        if msg["role"].as_str() == Some("assistant") {
            if let Some(content) = msg["content"].as_array() {
                for block in content {
                    if block["type"].as_str() == Some("tool_use") {
                        if let (Some(id), Some(name)) =
                            (block["id"].as_str(), block["name"].as_str())
                        {
                            tool_name_map.insert(id.to_string(), name.to_string());
                        }
                    }
                }
            }
        }
    }

    // Clear old results (keep last KEEP_RECENT)
    let to_clear_count = tool_results.len() - KEEP_RECENT;
    for tr_ref in &tool_results[..to_clear_count] {
        let content_str = messages[tr_ref.msg_idx]["content"]
            .as_array()
            .and_then(|arr| arr.get(tr_ref.part_idx))
            .and_then(|part| part["content"].as_str())
            .unwrap_or("");

        if content_str.len() > 100 {
            let tool_id = messages[tr_ref.msg_idx]["content"]
                .as_array()
                .and_then(|arr| arr.get(tr_ref.part_idx))
                .and_then(|part| part["tool_use_id"].as_str())
                .unwrap_or("")
                .to_string();

            let tool_name = tool_name_map
                .get(&tool_id)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            let placeholder = format!("[Previous: used {}]", tool_name);

            // Update in place
            if let Some(content) = messages[tr_ref.msg_idx]["content"].as_array_mut() {
                if let Some(part) = content.get_mut(tr_ref.part_idx) {
                    part["content"] = serde_json::Value::String(placeholder);
                }
            }
        }
    }
}

/// Layer 2: auto_compact — save transcript, summarize, replace messages.
/// Returns (new_messages, transcript_path).
pub fn auto_compact(
    messages: &[serde_json::Value],
    llm: &mut dyn Llm,
    transcript_dir: &Path,
) -> (Vec<serde_json::Value>, PathBuf) {
    // Save full transcript
    std::fs::create_dir_all(transcript_dir).ok();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let transcript_path = transcript_dir.join(format!("transcript_{}.jsonl", timestamp));

    let mut content = String::new();
    for msg in messages {
        content.push_str(&serde_json::to_string(msg).unwrap_or_default());
        content.push('\n');
    }
    std::fs::write(&transcript_path, &content).ok();

    // Ask LLM to summarize
    let conversation_text = serde_json::to_string(messages).unwrap_or_default();
    let truncated = if conversation_text.len() > 80000 {
        &conversation_text[..80000]
    } else {
        &conversation_text
    };

    let response = llm.create(LlmParams {
        model: "test".to_string(),
        system: String::new(),
        messages: vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text(format!(
                "Summarize this conversation for continuity. Include: \
                 1) What was accomplished, 2) Current state, 3) Key decisions made. \
                 Be concise but preserve critical details.\n\n{}",
                truncated
            )),
        }],
        tools: Vec::new(),
        max_tokens: 2000,
    });

    let summary = response
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");

    let new_messages = vec![
        serde_json::json!({
            "role": "user",
            "content": format!("[Conversation compressed. Transcript: {}]\n\n{}", transcript_path.display(), summary),
        }),
        serde_json::json!({
            "role": "assistant",
            "content": "Understood. I have the context from the summary. Continuing.",
        }),
    ];

    (new_messages, transcript_path)
}

/// TranscriptStore: save and list transcripts.
pub struct TranscriptStore {
    pub directory: PathBuf,
}

impl TranscriptStore {
    pub fn new(directory: &Path) -> Self {
        std::fs::create_dir_all(directory).ok();
        Self {
            directory: directory.to_path_buf(),
        }
    }

    pub fn save(&self, messages: &[serde_json::Value]) -> PathBuf {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let path = self.directory.join(format!("transcript_{}.jsonl", timestamp));
        let mut content = String::new();
        for msg in messages {
            content.push_str(&serde_json::to_string(msg).unwrap_or_default());
            content.push('\n');
        }
        std::fs::write(&path, &content).ok();
        path
    }

    pub fn list(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.directory) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map_or(false, |n| n.starts_with("transcript_") && n.ends_with(".jsonl"))
                {
                    paths.push(path);
                }
            }
        }
        paths.sort();
        paths
    }
}

use crate::types::*;
use std::path::{Path, PathBuf};

pub const KEEP_RECENT: usize = 10;
pub const THRESHOLD: usize = 180_000;

/// Rough token count: ~4 chars per token.
pub fn estimate_tokens(messages: &[serde_json::Value]) -> usize {
    let s = serde_json::to_string(messages).unwrap_or_default();
    s.len() / 4
}

/// Extract all text content from an LlmResponse, joining blocks.
pub fn extract_llm_text(resp: &LlmResponse) -> String {
    resp.content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Ask the LLM to summarize text, returning the summary string.
/// On failure, returns the fallback string.
fn summarize_via_llm(llm: &mut dyn Llm, prompt: &str, max_tokens: usize, fallback: &str) -> String {
    match llm.create(LlmParams {
        model: "default".to_string(),
        system: String::new(),
        messages: vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text(prompt.to_string()),
        }],
        tools: Vec::new(),
        max_tokens,
    }) {
        Ok(resp) => extract_llm_text(&resp),
        Err(e) => {
            eprintln!("[compact] Summarization error: {}", e);
            fallback.to_string()
        }
    }
}

/// Layer 1: micro_compact — replace old tool_results with placeholders.
/// Preserves the last KEEP_RECENT tool_results.
pub fn micro_compact(messages: &mut [serde_json::Value]) {
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
    let store = TranscriptStore::new(transcript_dir);
    let transcript_path = store.save(messages);

    let conversation_text = serde_json::to_string(messages).unwrap_or_default();
    let truncated = crate::util::truncate_at_boundary(&conversation_text, 300_000);

    let prompt = format!(
        "Summarize this conversation for continuity. Include: \
         1) What was accomplished, 2) Current state, 3) Key decisions made. \
         Be concise but preserve critical details.\n\n{}",
        truncated
    );
    let summary = summarize_via_llm(llm, &prompt, 4000, "");

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

// ---------------------------------------------------------------------------
// GAP 1: ContextGuard helpers — typed truncation and compaction
// ---------------------------------------------------------------------------

/// Truncate tool_result content blocks that exceed max_len characters.
/// Works on typed Message vec. Returns true if any truncation occurred.
pub fn truncate_tool_results(messages: &mut [Message], max_len: usize) -> bool {
    let mut truncated = false;
    for msg in messages.iter_mut() {
        if msg.role != "user" {
            continue;
        }
        if let MessageContent::Blocks(ref mut blocks) = msg.content {
            for block in blocks.iter_mut() {
                if let ContentBlock::ToolResult {
                    ref mut content, ..
                } = block
                {
                    if content.len() > max_len {
                        let boundary = crate::util::truncate_at_boundary(content, max_len).len();
                        content.truncate(boundary);
                        content.push_str("\n... [truncated by context guard]");
                        truncated = true;
                    }
                }
            }
        }
    }
    truncated
}

/// Compact messages for overflow recovery: summarize the first half, keep the recent half.
/// Returns a new message vec with ~50% fewer tokens.
pub fn compact_for_overflow(
    messages: &[Message],
    llm: &mut dyn Llm,
    transcript_dir: &Path,
) -> Vec<Message> {
    // Save transcript
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| serde_json::to_value(m).unwrap_or_default())
        .collect();
    let store = TranscriptStore::new(transcript_dir);
    let transcript_path = store.save(&json_messages);

    // Split: summarize first half, keep second half
    let mid = messages.len() / 2;
    let (old, recent) = messages.split_at(mid.max(1));

    let old_text = old
        .iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    let truncated_old = crate::util::truncate_at_boundary(&old_text, 200_000);

    let prompt = format!(
        "Compress this conversation history into a brief summary. \
         Preserve: tool calls made, key results, decisions, and current state.\n\n{}",
        truncated_old
    );
    let fallback = format!(
        "[Earlier conversation truncated. Transcript: {}]",
        transcript_path.display()
    );
    let summary = summarize_via_llm(llm, &prompt, 2000, &fallback);

    let mut result = vec![
        Message {
            role: "user".to_string(),
            content: MessageContent::Text(format!(
                "[Context compressed. Transcript: {}]\n\n{}",
                transcript_path.display(),
                summary
            )),
        },
        Message {
            role: "assistant".to_string(),
            content: MessageContent::Text(
                "Understood. I have the compressed context. Continuing.".to_string(),
            ),
        },
    ];
    result.extend_from_slice(recent);
    result
}

// ---------------------------------------------------------------------------
// TranscriptStore: save and list transcripts
// ---------------------------------------------------------------------------

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
        let path = self
            .directory
            .join(format!("transcript_{}.jsonl", timestamp));
        let mut content = String::new();
        for msg in messages {
            content.push_str(&serde_json::to_string(msg).unwrap_or_default());
            content.push('\n');
        }
        std::fs::write(&path, &content).ok();
        path
    }

    pub fn list(&self) -> Vec<PathBuf> {
        crate::util::list_files_matching(&self.directory, "transcript_", ".jsonl")
    }
}

// ---------------------------------------------------------------------------
// GAP 2: SessionStore — JSONL-based session replay
// ---------------------------------------------------------------------------

pub struct SessionStore {
    path: PathBuf,
    pub session_id: String,
}

impl SessionStore {
    /// Create or open a session file.
    pub fn new(session_dir: &Path, session_id: &str) -> Self {
        std::fs::create_dir_all(session_dir).ok();
        let path = session_dir.join(format!("session_{}.jsonl", session_id));
        Self {
            path,
            session_id: session_id.to_string(),
        }
    }

    /// Append new messages to the session JSONL file.
    pub fn append_turn(&self, new_messages: &[serde_json::Value]) {
        use std::io::Write;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            for msg in new_messages {
                let entry = serde_json::json!({
                    "ts": ts,
                    "msg": msg,
                });
                let line = serde_json::to_string(&entry).unwrap_or_default();
                writeln!(file, "{}", line).ok();
            }
        }
    }

    /// Rebuild the full message history from the session JSONL file.
    pub fn rebuild(&self) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();
        if let Ok(content) = std::fs::read_to_string(&self.path) {
            for line in content.lines() {
                if let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(msg) = entry.get("msg") {
                        messages.push(msg.clone());
                    }
                }
            }
        }
        messages
    }

    /// List all available sessions in a directory.
    pub fn list_sessions(session_dir: &Path) -> Vec<(String, PathBuf)> {
        let paths = crate::util::list_files_matching(session_dir, "session_", ".jsonl");
        paths
            .into_iter()
            .filter_map(|path| {
                let name = path.file_name()?.to_str()?;
                let id = name.strip_prefix("session_")?.strip_suffix(".jsonl")?;
                Some((id.to_string(), path))
            })
            .collect()
    }
}

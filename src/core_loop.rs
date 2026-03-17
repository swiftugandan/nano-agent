use crate::types::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Run the core agent loop pattern with an LLM.
/// Returns the number of LLM calls made.
///
/// `signals` bundles optional control signals:
/// - `compact_signal` + `transcript_dir`: on-demand compaction when the LLM calls `compact`
/// - `idle_signal`: breaks the loop when the agent calls `idle` (for teammate lifecycle)
pub fn run_agent_loop(
    llm: &mut dyn Llm,
    system: &str,
    messages: &mut Vec<serde_json::Value>,
    tools: &[serde_json::Value],
    dispatch: &Dispatch,
    signals: &LoopSignals,
) -> usize {
    let mut call_count = 0;
    loop {
        // Check interrupt before LLM call
        if let Some(sig) = signals.interrupt_signal {
            if sig.load(Ordering::Acquire) {
                break;
            }
        }

        let typed_messages: Vec<Message> = messages
            .iter()
            .filter_map(|m| serde_json::from_value(m.clone()).ok())
            .collect();

        let response = match llm.create(LlmParams {
            model: "default".to_string(),
            system: system.to_string(),
            messages: typed_messages,
            tools: tools.to_vec(),
            max_tokens: 16_000,
        }) {
            Ok(r) => r,
            Err(LlmError::Overflow { message }) => {
                // Attempt compaction and continue
                if let (Some(signal), Some(_dir)) = (signals.compact_signal, signals.transcript_dir)
                {
                    signal.request();
                }
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": format!("[System: Context overflow — {}. Compaction requested.]", message),
                }));
                continue;
            }
            Err(e) => {
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": format!("[System: LLM error — {}. Stopping.]", e),
                }));
                break;
            }
        };
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
        let mut interrupted = false;
        for block in &response.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                // Check interrupt before each tool execution
                if let Some(sig) = signals.interrupt_signal {
                    if sig.load(Ordering::Acquire) {
                        results.push(serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": id,
                            "content": "[Interrupted by user]",
                        }));
                        interrupted = true;
                        continue;
                    }
                }

                // Emit Start event
                if let Some(cb) = signals.tool_callback {
                    cb(ToolEvent::Start {
                        name: name.clone(),
                        input: input.clone(),
                    });
                }

                let start = Instant::now();
                let result = route(dispatch, name, input.clone());
                let duration = start.elapsed();

                let output = match &result {
                    Ok(out) => out.clone(),
                    Err(err) => err.clone(),
                };

                // Emit Complete or Error event
                if let Some(cb) = signals.tool_callback {
                    match &result {
                        Err(err) => {
                            cb(ToolEvent::Error {
                                name: name.clone(),
                                error: err.clone(),
                                duration,
                            });
                        }
                        Ok(out) => {
                            cb(ToolEvent::Complete {
                                name: name.clone(),
                                summary: summarize_tool_output(out),
                                duration,
                            });
                        }
                    }
                }

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

        if interrupted {
            break;
        }

        // Check compact signal: run auto_compact if requested
        if let (Some(signal), Some(dir)) = (signals.compact_signal, signals.transcript_dir) {
            if signal.take() {
                let (new_msgs, path) = crate::memory::auto_compact(messages, llm, dir);
                *messages = new_msgs;
                eprintln!("[compact] On-demand compaction saved: {}", path.display());
            }
        }

        // Check idle signal: break loop to return control to lifecycle
        if let Some(signal) = signals.idle_signal {
            if signal.load(Ordering::Acquire) {
                break;
            }
        }
    }
    call_count
}

/// Route a tool call to the correct handler. Never panics.
/// Returns `Ok(output)` on success, `Err(message)` for unknown tools or
/// handlers that return an error string (prefixed with `"Error: "`).
pub fn route(
    dispatch: &Dispatch,
    tool_name: &str,
    input: serde_json::Value,
) -> Result<String, String> {
    match dispatch.get(tool_name) {
        Some(handler) => {
            let output = handler(input);
            if output.starts_with("Error: ") {
                Err(output)
            } else {
                Ok(output)
            }
        }
        None => Err(format!("Unknown tool: {}", tool_name)),
    }
}

/// Summarize tool output for display (line count or truncated text).
fn summarize_tool_output(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() > 3 {
        format!("{} lines", lines.len())
    } else if output.chars().count() > 80 {
        let truncated: String = output.chars().take(77).collect();
        format!("{}...", truncated)
    } else {
        output.to_string()
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

use crate::handler::{AgentContext, HandlerRegistry};
use crate::types::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Run the core agent loop pattern with an LLM.
/// Returns the number of LLM calls made.
pub fn run_agent_loop(
    llm: &mut dyn Llm,
    system: &str,
    messages: &mut Vec<serde_json::Value>,
    tools: &[serde_json::Value],
    registry: &HandlerRegistry,
    ctx: &AgentContext,
) -> usize {
    let mut call_count = 0;
    loop {
        // Check interrupt before LLM call
        if let Some(ref sig) = ctx.signals.interrupt {
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
                ctx.signals.compact.request();
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
                if let Some(ref sig) = ctx.signals.interrupt {
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
                if let Some(ref cb) = ctx.tool_callback {
                    cb(ToolEvent::Start {
                        name: name.clone(),
                        input: input.clone(),
                    });
                }

                let start = Instant::now();
                let result = registry.route(ctx, name, input.clone());
                let duration = start.elapsed();

                let output = match &result {
                    Ok(out) => out.clone(),
                    Err(e) => format!("{}", e),
                };

                // Emit Complete or Error event
                if let Some(ref cb) = ctx.tool_callback {
                    match &result {
                        Err(e) => {
                            cb(ToolEvent::Error {
                                name: name.clone(),
                                error: format!("{}", e),
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

                let projected = if let Some(ref proj) = ctx.projector {
                    proj.project("tool_results", id, &output)
                } else {
                    output
                };
                results.push(serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": projected,
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
        if let Some(ref dir) = ctx.transcript_dir {
            if ctx.signals.compact.take() {
                let (new_msgs, path) = crate::memory::auto_compact(messages, llm, dir);
                *messages = new_msgs;
                eprintln!("[compact] On-demand compaction saved: {}", path.display());
            }
        }

        // Check idle signal: break loop to return control to lifecycle
        if ctx.signals.idle.load(Ordering::Acquire) {
            break;
        }
    }
    call_count
}

/// Summarize tool output for display (line count or truncated text).
fn summarize_tool_output(output: &str) -> String {
    let line_count = output.lines().count();
    if line_count > 3 {
        format!("{} lines", line_count)
    } else if output.chars().count() > 80 {
        let truncated: String = output.chars().take(77).collect();
        format!("{}...", truncated)
    } else {
        output.to_string()
    }
}

/// Path sandbox: ensures paths stay within the workspace.
#[derive(Clone)]
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

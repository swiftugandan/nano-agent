use crate::core_loop::PathSandbox;
use crate::handler::{
    exec_err, require_str, with_output_cap, AgentContext, Chain, Handler, HandlerError,
    HandlerRegistry, HandlerResult,
};
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

/// Return the JSON schema definitions for the built-in tools.
/// Cached after first call to avoid rebuilding static JSON each time.
pub fn tool_definitions() -> Vec<serde_json::Value> {
    static DEFS: OnceLock<Vec<serde_json::Value>> = OnceLock::new();
    DEFS.get_or_init(tool_definitions_inner).clone()
}

fn tool_definitions_inner() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "name": "bash",
            "description": "Execute a shell command and return stdout+stderr.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds. Process is killed after this duration."
                    }
                },
                "required": ["command"]
            }
        }),
        serde_json::json!({
            "name": "read_file",
            "description": "Read a file's contents. Optionally start from a given line and/or limit the number of lines.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of lines to return (optional)"
                    }
                },
                "required": ["path"]
            }
        }),
        serde_json::json!({
            "name": "write_file",
            "description": "Write content to a file, creating parent directories as needed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write"
                    }
                },
                "required": ["path", "content"]
            }
        }),
        serde_json::json!({
            "name": "edit_file",
            "description": "Replace the first occurrence of old_text with new_text in a file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Text to find"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text"
                    }
                },
                "required": ["path", "old_text", "new_text"]
            }
        }),
        serde_json::json!({
            "name": "grep",
            "description": "Search file contents using ripgrep. Returns matching lines with file paths and line numbers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: workspace root)"
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.rs')"
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive search"
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "Treat pattern as a literal string, not regex"
                    },
                    "context": {
                        "type": "integer",
                        "description": "Number of context lines before and after each match"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of matching lines to return (default: 100)"
                    }
                },
                "required": ["pattern"]
            }
        }),
        serde_json::json!({
            "name": "find",
            "description": "Find files by glob pattern using fd.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match file names (e.g. '*.rs')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: workspace root)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results (default: 1000)"
                    }
                },
                "required": ["pattern"]
            }
        }),
        serde_json::json!({
            "name": "ls",
            "description": "List directory contents. Directories are shown with a trailing '/'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: workspace root)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of entries to return (default: 500)"
                    }
                }
            }
        }),
    ]
}

// Safety blocklist for bash commands
const BLOCKED_COMMANDS: &[&str] = &[
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){",
    "fork bomb",
];

fn format_command_output(stdout: &str, stderr: &str, exit_code: Option<i32>) -> String {
    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str("[stderr] ");
        result.push_str(stderr);
    }
    if result.is_empty() {
        result = format!("(exit code {})", exit_code.unwrap_or(-1));
    }
    result
}

fn resolve_path(
    input: &serde_json::Value,
    sandbox: &PathSandbox,
    workspace: &Path,
) -> Result<PathBuf, String> {
    match input.get("path").and_then(|v| v.as_str()) {
        Some(p) => sandbox.safe_path(p).map_err(|e| format!("{}", e)),
        None => Ok(workspace.to_path_buf()),
    }
}

fn format_external_output(output: Output, empty_msg: &str) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stdout.is_empty() && !stderr.is_empty() {
        return format!("Error: {}", stderr.trim());
    }
    if stdout.is_empty() {
        return empty_msg.to_string();
    }
    stdout.into_owned()
}

// ---------------------------------------------------------------------------
// Handler structs for the 7 core tools
// ---------------------------------------------------------------------------

pub struct BashHandler {
    pub workspace: PathBuf,
}

impl Handler for BashHandler {
    fn call(&self, _ctx: &AgentContext, input: serde_json::Value) -> HandlerResult {
        let cmd = require_str(&input, "command")?;

        for blocked in BLOCKED_COMMANDS {
            if cmd.contains(blocked) {
                return Err(HandlerError::Validation {
                    message: format!("command blocked for safety: {}", blocked),
                });
            }
        }

        let timeout_secs = input.get("timeout").and_then(|v| v.as_f64());

        let mut child = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(&self.workspace)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| HandlerError::Execution {
                message: format!("Error spawning command: {}", e),
            })?;

        match timeout_secs {
            Some(secs) => {
                let deadline = Instant::now() + Duration::from_secs_f64(secs);

                let stdout_pipe = child.stdout.take();
                let stderr_pipe = child.stderr.take();

                let stdout_handle = thread::spawn(move || {
                    let mut buf = Vec::new();
                    if let Some(mut pipe) = stdout_pipe {
                        let _ = pipe.read_to_end(&mut buf);
                    }
                    buf
                });

                let stderr_handle = thread::spawn(move || {
                    let mut buf = Vec::new();
                    if let Some(mut pipe) = stderr_pipe {
                        let _ = pipe.read_to_end(&mut buf);
                    }
                    buf
                });

                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            let stdout_data = stdout_handle.join().unwrap_or_default();
                            let stderr_data = stderr_handle.join().unwrap_or_default();
                            let stdout = String::from_utf8_lossy(&stdout_data);
                            let stderr = String::from_utf8_lossy(&stderr_data);
                            return Ok(format_command_output(&stdout, &stderr, status.code()));
                        }
                        Ok(None) => {
                            if Instant::now() >= deadline {
                                let _ = child.kill();
                                let _ = child.wait();
                                let stdout_data = stdout_handle.join().unwrap_or_default();
                                let stderr_data = stderr_handle.join().unwrap_or_default();
                                let stdout = String::from_utf8_lossy(&stdout_data);
                                let stderr = String::from_utf8_lossy(&stderr_data);
                                let mut msg = format!("command timed out after {}s", secs);
                                if !stdout.is_empty() {
                                    msg.push_str("\n[stdout] ");
                                    msg.push_str(&stdout);
                                }
                                if !stderr.is_empty() {
                                    msg.push_str("\n[stderr] ");
                                    msg.push_str(&stderr);
                                }
                                return Err(HandlerError::Execution { message: msg });
                            }
                            thread::sleep(Duration::from_millis(50));
                        }
                        Err(e) => {
                            return Err(HandlerError::Execution {
                                message: format!("Error waiting for process: {}", e),
                            });
                        }
                    }
                }
            }
            None => match child.wait_with_output() {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    Ok(format_command_output(
                        &stdout,
                        &stderr,
                        output.status.code(),
                    ))
                }
                Err(e) => Err(HandlerError::Execution {
                    message: format!("Error waiting for process: {}", e),
                }),
            },
        }
    }
}

pub struct ReadFileHandler {
    pub sandbox: Arc<PathSandbox>,
}

impl Handler for ReadFileHandler {
    fn call(&self, _ctx: &AgentContext, input: serde_json::Value) -> HandlerResult {
        let path_str = require_str(&input, "path")?;
        let resolved = self.sandbox.safe_path(path_str).map_err(exec_err)?;
        match std::fs::read_to_string(&resolved) {
            Ok(contents) => {
                let offset = input
                    .get("offset")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1)
                    .max(1) as usize;
                let limit = input
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize)
                    .unwrap_or(usize::MAX);
                Ok(contents
                    .lines()
                    .skip(offset - 1)
                    .take(limit)
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            Err(e) => Err(exec_err(e)),
        }
    }
}

pub struct WriteFileHandler {
    pub sandbox: Arc<PathSandbox>,
}

impl Handler for WriteFileHandler {
    fn call(&self, _ctx: &AgentContext, input: serde_json::Value) -> HandlerResult {
        let path_str = require_str(&input, "path")?;
        let content = require_str(&input, "content")?;
        let resolved = self.sandbox.safe_path(path_str).map_err(exec_err)?;
        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent).map_err(exec_err)?;
        }
        std::fs::write(&resolved, content).map_err(exec_err)?;
        Ok(format!("Wrote {} bytes to {}", content.len(), path_str))
    }
}

pub struct EditFileHandler {
    pub sandbox: Arc<PathSandbox>,
}

impl Handler for EditFileHandler {
    fn call(&self, _ctx: &AgentContext, input: serde_json::Value) -> HandlerResult {
        let path_str = require_str(&input, "path")?;
        let old_text = require_str(&input, "old_text")?;
        let new_text = require_str(&input, "new_text")?;
        let resolved = self.sandbox.safe_path(path_str).map_err(exec_err)?;
        let contents = std::fs::read_to_string(&resolved).map_err(exec_err)?;
        if !contents.contains(old_text) {
            return Err(HandlerError::Execution {
                message: format!("old_text not found in {}", path_str),
            });
        }
        let updated = contents.replacen(old_text, new_text, 1);
        std::fs::write(&resolved, &updated).map_err(exec_err)?;
        Ok(format!("Edited {}", path_str))
    }
}

pub struct GrepHandler {
    pub sandbox: Arc<PathSandbox>,
    pub workspace: PathBuf,
}

impl Handler for GrepHandler {
    fn call(&self, _ctx: &AgentContext, input: serde_json::Value) -> HandlerResult {
        let pattern = require_str(&input, "pattern")?;

        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(100);

        let search_dir = resolve_path(&input, &self.sandbox, &self.workspace).map_err(exec_err)?;

        let mut cmd = Command::new("rg");
        cmd.arg("--no-heading")
            .arg("--line-number")
            .arg("--color=never");

        if input
            .get("ignore_case")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            cmd.arg("-i");
        }
        if input
            .get("literal")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            cmd.arg("--fixed-strings");
        }
        if let Some(context_lines) = input.get("context").and_then(|v| v.as_u64()) {
            cmd.arg("-C").arg(context_lines.to_string());
        }
        if let Some(glob) = input.get("glob").and_then(|v| v.as_str()) {
            cmd.arg("--glob").arg(glob);
        }

        cmd.arg("--").arg(pattern).arg(&search_dir);

        match cmd.output() {
            Ok(output) => {
                let result = format_external_output(output, "No matches found.");
                let limit_usize = limit as usize;
                let lines: Vec<&str> = result.lines().collect();
                if lines.len() > limit_usize {
                    let truncated = lines[..limit_usize].join("\n");
                    Ok(format!("{}\n... (truncated to {} lines)", truncated, limit))
                } else {
                    Ok(result)
                }
            }
            Err(e) => Err(HandlerError::Execution {
                message: format!("Error running rg: {}", e),
            }),
        }
    }
}

pub struct FindHandler {
    pub sandbox: Arc<PathSandbox>,
    pub workspace: PathBuf,
}

impl Handler for FindHandler {
    fn call(&self, _ctx: &AgentContext, input: serde_json::Value) -> HandlerResult {
        let pattern = require_str(&input, "pattern")?;

        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(1000);

        let search_dir = resolve_path(&input, &self.sandbox, &self.workspace).map_err(exec_err)?;

        let output = Command::new("fd")
            .arg("--glob")
            .arg(pattern)
            .arg("--color=never")
            .arg("--max-results")
            .arg(limit.to_string())
            .arg(&search_dir)
            .output();

        match output {
            Ok(output) => Ok(format_external_output(output, "No files found.")),
            Err(e) => Err(HandlerError::Execution {
                message: format!("Error running fd: {}", e),
            }),
        }
    }
}

pub struct LsHandler {
    pub sandbox: Arc<PathSandbox>,
    pub workspace: PathBuf,
}

impl Handler for LsHandler {
    fn call(&self, _ctx: &AgentContext, input: serde_json::Value) -> HandlerResult {
        let dir = resolve_path(&input, &self.sandbox, &self.workspace).map_err(exec_err)?;

        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(500) as usize;

        let entries = std::fs::read_dir(&dir).map_err(exec_err)?;

        let mut names: Vec<String> = Vec::new();
        for entry in entries {
            match entry {
                Ok(e) => {
                    let mut name = e.file_name().to_string_lossy().to_string();
                    if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        name.push('/');
                    }
                    names.push(name);
                }
                Err(_) => continue,
            }
        }

        names.sort();
        names.truncate(limit);
        if names.is_empty() {
            Ok("(empty directory)".to_string())
        } else {
            Ok(names.join("\n"))
        }
    }
}

// ---------------------------------------------------------------------------
// Build registry (replaces build_dispatch)
// ---------------------------------------------------------------------------

/// Build a handler registry with real tool handlers rooted at the given workspace.
pub fn build_registry(workspace: &Path) -> HandlerRegistry {
    let mut reg = HandlerRegistry::new();
    let sandbox = Arc::new(PathSandbox::new(workspace));

    reg.register(
        "bash",
        Chain::new(Arc::new(BashHandler {
            workspace: workspace.to_path_buf(),
        }))
        .with(with_output_cap(50 * 1024))
        .build(),
    );

    reg.register(
        "read_file",
        Chain::new(Arc::new(ReadFileHandler {
            sandbox: Arc::clone(&sandbox),
        }))
        .with(with_output_cap(100 * 1024))
        .build(),
    );

    reg.register(
        "write_file",
        Arc::new(WriteFileHandler {
            sandbox: Arc::clone(&sandbox),
        }),
    );

    reg.register(
        "edit_file",
        Arc::new(EditFileHandler {
            sandbox: Arc::clone(&sandbox),
        }),
    );

    reg.register(
        "grep",
        Chain::new(Arc::new(GrepHandler {
            sandbox: Arc::clone(&sandbox),
            workspace: workspace.to_path_buf(),
        }))
        .with(with_output_cap(50 * 1024))
        .build(),
    );

    reg.register(
        "find",
        Chain::new(Arc::new(FindHandler {
            sandbox: Arc::clone(&sandbox),
            workspace: workspace.to_path_buf(),
        }))
        .with(with_output_cap(50 * 1024))
        .build(),
    );

    reg.register(
        "ls",
        Arc::new(LsHandler {
            sandbox,
            workspace: workspace.to_path_buf(),
        }),
    );

    reg
}

// ---------------------------------------------------------------------------
// Extended tool definitions (L2–L11)
// ---------------------------------------------------------------------------

/// Return extended tool definitions (L2–L11).
/// Cached after first call to avoid rebuilding static JSON each time.
pub fn extended_tool_definitions() -> Vec<serde_json::Value> {
    static DEFS: OnceLock<Vec<serde_json::Value>> = OnceLock::new();
    DEFS.get_or_init(extended_tool_definitions_inner).clone()
}

fn extended_tool_definitions_inner() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "name": "todo_update",
            "description": "Update the todo list. Only one item may be in_progress at a time.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":     { "type": "string" },
                                "text":   { "type": "string" },
                                "status": { "type": "string", "enum": ["pending","in_progress","completed"] }
                            },
                            "required": ["id","text","status"]
                        }
                    }
                },
                "required": ["items"]
            }
        }),
        serde_json::json!({
            "name": "todo_read",
            "description": "Read the current todo list.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "read_skill",
            "description": "Load a skill by name and return its full content.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Skill name" }
                },
                "required": ["name"]
            }
        }),
        serde_json::json!({
            "name": "task_create",
            "description": "Create a new task with a subject.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string" }
                },
                "required": ["subject"]
            }
        }),
        serde_json::json!({
            "name": "task_get",
            "description": "Get a task by ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "integer" }
                },
                "required": ["task_id"]
            }
        }),
        serde_json::json!({
            "name": "task_update",
            "description": "Update a task's status or dependencies.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id":        { "type": "integer" },
                    "status":         { "type": "string", "enum": ["pending","in_progress","completed"] },
                    "add_blocked_by": { "type": "array", "items": { "type": "integer" } },
                    "add_blocks":     { "type": "array", "items": { "type": "integer" } }
                },
                "required": ["task_id"]
            }
        }),
        serde_json::json!({
            "name": "task_list",
            "description": "List all tasks with their status and dependencies.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "background_run",
            "description": "Run a shell command in the background. Returns a task ID immediately.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "lane": { "type": "string", "description": "Lane name (default: 'background'). Available: main, cron, background" }
                },
                "required": ["command"]
            }
        }),
        serde_json::json!({
            "name": "background_check",
            "description": "Check the status and result of a background task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" }
                },
                "required": ["task_id"]
            }
        }),
        serde_json::json!({
            "name": "send_message",
            "description": "Send a message to a teammate via the message bus.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to":      { "type": "string", "description": "Recipient name" },
                    "content": { "type": "string" }
                },
                "required": ["to","content"]
            }
        }),
        serde_json::json!({
            "name": "broadcast_message",
            "description": "Broadcast a message to all teammates.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"]
            }
        }),
        serde_json::json!({
            "name": "read_inbox",
            "description": "Read and clear your inbox messages.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "list_teammates",
            "description": "List all team members and their roles.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "spawn_teammate",
            "description": "Create or reactivate a teammate.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":   { "type": "string" },
                    "role":   { "type": "string" },
                    "prompt": { "type": "string" }
                },
                "required": ["name","role","prompt"]
            }
        }),
        serde_json::json!({
            "name": "shutdown_teammate",
            "description": "Request a teammate to shut down (requires their approval).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "teammate": { "type": "string" }
                },
                "required": ["teammate"]
            }
        }),
        serde_json::json!({
            "name": "review_plan",
            "description": "Approve or reject a pending plan request.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "request_id": { "type": "string" },
                    "approve":    { "type": "boolean" },
                    "feedback":   { "type": "string" }
                },
                "required": ["request_id","approve","feedback"]
            }
        }),
        serde_json::json!({
            "name": "worktree_create",
            "description": "Create a git worktree, optionally bound to a task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":    { "type": "string", "description": "Worktree name (letters, numbers, .-_)" },
                    "task_id": { "type": "integer", "description": "Optional task to bind" }
                },
                "required": ["name"]
            }
        }),
        serde_json::json!({
            "name": "worktree_remove",
            "description": "Remove a git worktree.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":          { "type": "string" },
                    "complete_task": { "type": "boolean", "description": "Mark bound task as completed" }
                },
                "required": ["name"]
            }
        }),
        serde_json::json!({
            "name": "list_events",
            "description": "List recent events from the event log.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "description": "Number of events (default 20)" }
                }
            }
        }),
        serde_json::json!({
            "name": "scan_tasks",
            "description": "Scan for unclaimed tasks (pending, no owner, not blocked).",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "claim_task",
            "description": "Claim an unclaimed task, setting yourself as owner.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "integer" }
                },
                "required": ["task_id"]
            }
        }),
        serde_json::json!({
            "name": "subagent",
            "description": "Spawn an isolated subagent to handle a subtask. Returns only its final text output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "Task description for the subagent" }
                },
                "required": ["prompt"]
            }
        }),
        serde_json::json!({
            "name": "compact",
            "description": "Trigger on-demand conversation compaction. Saves the current transcript and replaces messages with a summary. Use when the conversation is getting long and you want to free up context.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "idle",
            "description": "Transition from WORK to IDLE phase. Use when you have completed your current task and are waiting for new work. Only available to teammates, not the lead agent.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "cron_add",
            "description": "Add a scheduled cron job. The prompt will be injected as a user message when the cron fires.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":   { "type": "string", "description": "Unique name for this cron entry" },
                    "cron":   { "type": "string", "description": "Cron expression: minute hour day month weekday (e.g. '*/5 * * * *')" },
                    "prompt": { "type": "string", "description": "Prompt to inject when cron fires" }
                },
                "required": ["name", "cron", "prompt"]
            }
        }),
        serde_json::json!({
            "name": "cron_remove",
            "description": "Remove a scheduled cron job by name.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                },
                "required": ["name"]
            }
        }),
        serde_json::json!({
            "name": "cron_list",
            "description": "List all cron entries with their schedules and enabled status.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "save_memory",
            "description": "Save a memory for future recall. Memories are searchable via TF-IDF.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The memory content to save" },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional tags for categorization"
                    }
                },
                "required": ["text"]
            }
        }),
        serde_json::json!({
            "name": "enqueue_delivery",
            "description": "Enqueue a message for reliable delivery to a channel+peer. Retries with exponential backoff.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel":      { "type": "string", "description": "Target channel name (e.g. 'cli', 'websocket')" },
                    "peer_id":      { "type": "string", "description": "Target peer identifier" },
                    "payload":      { "type": "string", "description": "Message payload to deliver" },
                    "max_attempts": { "type": "integer", "description": "Max delivery attempts (default 5)" }
                },
                "required": ["channel", "peer_id", "payload"]
            }
        }),
    ]
}

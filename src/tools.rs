use crate::core_loop::PathSandbox;
use crate::types::Dispatch;
use std::collections::HashMap;
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, Instant};

/// Return the JSON schema definitions for the built-in tools.
pub fn tool_definitions() -> Vec<serde_json::Value> {
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

const MAX_OUTPUT_BYTES: usize = 50 * 1024; // 50KB

fn truncate_output(s: String) -> String {
    if s.len() > MAX_OUTPUT_BYTES {
        let mut end = MAX_OUTPUT_BYTES;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        let mut truncated = s[..end].to_string();
        truncated.push_str("\n... (output truncated)");
        truncated
    } else {
        s
    }
}

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
        Some(p) => sandbox.safe_path(p).map_err(|e| format!("Error: {}", e)),
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
    truncate_output(stdout.into_owned())
}

/// Build a dispatch table with real tool handlers rooted at the given workspace.
pub fn build_dispatch(workspace: &Path) -> Dispatch {
    let mut dispatch: Dispatch = HashMap::new();

    // bash
    let ws = workspace.to_path_buf();
    dispatch.insert(
        "bash".to_string(),
        Box::new(move |input| {
            let cmd = match input.get("command").and_then(|v| v.as_str()) {
                Some(c) => c,
                None => return "Error: missing 'command' field".to_string(),
            };

            for blocked in BLOCKED_COMMANDS {
                if cmd.contains(blocked) {
                    return format!("Error: command blocked for safety: {}", blocked);
                }
            }

            let timeout_secs = input.get("timeout").and_then(|v| v.as_f64());

            let mut child = match Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(&ws)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => return format!("Error spawning command: {}", e),
            };

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

                    // Poll for exit or timeout
                    loop {
                        match child.try_wait() {
                            Ok(Some(status)) => {
                                let stdout_data = stdout_handle.join().unwrap_or_default();
                                let stderr_data = stderr_handle.join().unwrap_or_default();
                                let stdout = String::from_utf8_lossy(&stdout_data);
                                let stderr = String::from_utf8_lossy(&stderr_data);
                                return format_command_output(&stdout, &stderr, status.code());
                            }
                            Ok(None) => {
                                if Instant::now() >= deadline {
                                    let _ = child.kill();
                                    let _ = child.wait();
                                    let stdout_data = stdout_handle.join().unwrap_or_default();
                                    let stderr_data = stderr_handle.join().unwrap_or_default();
                                    let stdout = String::from_utf8_lossy(&stdout_data);
                                    let stderr = String::from_utf8_lossy(&stderr_data);
                                    let mut result = format!("Error: command timed out after {}s", secs);
                                    if !stdout.is_empty() {
                                        result.push_str("\n[stdout] ");
                                        result.push_str(&stdout);
                                    }
                                    if !stderr.is_empty() {
                                        result.push_str("\n[stderr] ");
                                        result.push_str(&stderr);
                                    }
                                    return result;
                                }
                                thread::sleep(Duration::from_millis(50));
                            }
                            Err(e) => return format!("Error waiting for process: {}", e),
                        }
                    }
                }
                None => {
                    match child.wait_with_output() {
                        Ok(output) => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            format_command_output(&stdout, &stderr, output.status.code())
                        }
                        Err(e) => format!("Error waiting for process: {}", e),
                    }
                }
            }
        }),
    );

    // read_file
    let sandbox_read = PathSandbox::new(workspace);
    dispatch.insert(
        "read_file".to_string(),
        Box::new(move |input| {
            let path_str = match input.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: missing 'path' field".to_string(),
            };
            let resolved = match sandbox_read.safe_path(path_str) {
                Ok(p) => p,
                Err(e) => return format!("Error: {}", e),
            };
            match std::fs::read_to_string(&resolved) {
                Ok(contents) => {
                    let offset = input.get("offset").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
                    let limit = input.get("limit").and_then(|v| v.as_u64()).map(|n| n as usize).unwrap_or(usize::MAX);
                    contents
                        .lines()
                        .skip(offset - 1)
                        .take(limit)
                        .collect::<Vec<_>>()
                        .join("\n")
                }
                Err(e) => format!("Error: {}", e),
            }
        }),
    );

    // write_file
    let sandbox_write = PathSandbox::new(workspace);
    dispatch.insert(
        "write_file".to_string(),
        Box::new(move |input| {
            let path_str = match input.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: missing 'path' field".to_string(),
            };
            let content = match input.get("content").and_then(|v| v.as_str()) {
                Some(c) => c,
                None => return "Error: missing 'content' field".to_string(),
            };
            let resolved = match sandbox_write.safe_path(path_str) {
                Ok(p) => p,
                Err(e) => return format!("Error: {}", e),
            };
            if let Some(parent) = resolved.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return format!("Error creating directories: {}", e);
                }
            }
            match std::fs::write(&resolved, content) {
                Ok(()) => format!("Wrote {} bytes to {}", content.len(), path_str),
                Err(e) => format!("Error: {}", e),
            }
        }),
    );

    // edit_file
    let sandbox_edit = PathSandbox::new(workspace);
    dispatch.insert(
        "edit_file".to_string(),
        Box::new(move |input| {
            let path_str = match input.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: missing 'path' field".to_string(),
            };
            let old_text = match input.get("old_text").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => return "Error: missing 'old_text' field".to_string(),
            };
            let new_text = match input.get("new_text").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => return "Error: missing 'new_text' field".to_string(),
            };
            let resolved = match sandbox_edit.safe_path(path_str) {
                Ok(p) => p,
                Err(e) => return format!("Error: {}", e),
            };
            match std::fs::read_to_string(&resolved) {
                Ok(contents) => {
                    if !contents.contains(old_text) {
                        return format!("Error: old_text not found in {}", path_str);
                    }
                    let updated = contents.replacen(old_text, new_text, 1);
                    match std::fs::write(&resolved, &updated) {
                        Ok(()) => format!("Edited {}", path_str),
                        Err(e) => format!("Error writing: {}", e),
                    }
                }
                Err(e) => format!("Error reading: {}", e),
            }
        }),
    );

    // grep (uses ripgrep)
    let sandbox_grep = PathSandbox::new(workspace);
    let ws_grep = workspace.to_path_buf();
    dispatch.insert(
        "grep".to_string(),
        Box::new(move |input| {
            let pattern = match input.get("pattern").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: missing 'pattern' field".to_string(),
            };

            let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(100);

            let search_dir = match resolve_path(&input, &sandbox_grep, &ws_grep) {
                Ok(p) => p,
                Err(e) => return e,
            };

            let mut cmd = Command::new("rg");
            cmd.arg("--no-heading")
                .arg("--line-number")
                .arg("--color=never");

            if input.get("ignore_case").and_then(|v| v.as_bool()).unwrap_or(false) {
                cmd.arg("-i");
            }
            if input.get("literal").and_then(|v| v.as_bool()).unwrap_or(false) {
                cmd.arg("--fixed-strings");
            }
            if let Some(ctx) = input.get("context").and_then(|v| v.as_u64()) {
                cmd.arg("-C").arg(ctx.to_string());
            }
            if let Some(glob) = input.get("glob").and_then(|v| v.as_str()) {
                cmd.arg("--glob").arg(glob);
            }

            cmd.arg("--").arg(pattern).arg(&search_dir);

            match cmd.output() {
                Ok(output) => {
                    let result = format_external_output(output, "No matches found.");
                    // Apply line limit: --max-count is per-file, so we limit total lines here
                    let lines: Vec<&str> = result.lines().take(limit as usize).collect();
                    if lines.len() < result.lines().count() {
                        format!("{}\n... (truncated to {} lines)", lines.join("\n"), limit)
                    } else {
                        result
                    }
                }
                Err(e) => format!("Error running rg: {}", e),
            }
        }),
    );

    // find (uses fd)
    let sandbox_find = PathSandbox::new(workspace);
    let ws_find = workspace.to_path_buf();
    dispatch.insert(
        "find".to_string(),
        Box::new(move |input| {
            let pattern = match input.get("pattern").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: missing 'pattern' field".to_string(),
            };

            let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(1000);

            let search_dir = match resolve_path(&input, &sandbox_find, &ws_find) {
                Ok(p) => p,
                Err(e) => return e,
            };

            let output = Command::new("fd")
                .arg("--glob")
                .arg(pattern)
                .arg("--color=never")
                .arg("--max-results")
                .arg(limit.to_string())
                .arg(&search_dir)
                .output();

            match output {
                Ok(output) => format_external_output(output, "No files found."),
                Err(e) => format!("Error running fd: {}", e),
            }
        }),
    );

    // ls
    let sandbox_ls = PathSandbox::new(workspace);
    let ws_ls = workspace.to_path_buf();
    dispatch.insert(
        "ls".to_string(),
        Box::new(move |input| {
            let dir = match resolve_path(&input, &sandbox_ls, &ws_ls) {
                Ok(p) => p,
                Err(e) => return e,
            };

            let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(500) as usize;

            let entries = match std::fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(e) => return format!("Error: {}", e),
            };

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
                "(empty directory)".to_string()
            } else {
                names.join("\n")
            }
        }),
    );

    dispatch
}

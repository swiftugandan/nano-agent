use crate::core_loop::PathSandbox;
use crate::types::Dispatch;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Return the JSON schema definitions for the four built-in tools.
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
                    }
                },
                "required": ["command"]
            }
        }),
        serde_json::json!({
            "name": "read_file",
            "description": "Read a file's contents. Optionally limit to first N lines.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file"
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

            match Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(&ws)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(child) => {
                    match child.wait_with_output() {
                        Ok(output) => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            let mut result = String::new();
                            if !stdout.is_empty() {
                                result.push_str(&stdout);
                            }
                            if !stderr.is_empty() {
                                if !result.is_empty() {
                                    result.push('\n');
                                }
                                result.push_str("[stderr] ");
                                result.push_str(&stderr);
                            }
                            if result.is_empty() {
                                result = format!("(exit code {})", output.status.code().unwrap_or(-1));
                            }
                            result
                        }
                        Err(e) => format!("Error waiting for process: {}", e),
                    }
                }
                Err(e) => format!("Error spawning command: {}", e),
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
                    if let Some(limit) = input.get("limit").and_then(|v| v.as_u64()) {
                        contents
                            .lines()
                            .take(limit as usize)
                            .collect::<Vec<_>>()
                            .join("\n")
                    } else {
                        contents
                    }
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

    dispatch
}

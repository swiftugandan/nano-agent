use std::collections::HashMap;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub status: String,
    pub result: String,
    pub command: String,
}

#[derive(Debug, Clone)]
pub struct Notification {
    pub task_id: String,
    pub status: String,
    pub command: String,
    pub result: String,
}

/// Truncate a string at a char boundary, avoiding panics on multi-byte UTF-8.
fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

pub struct BackgroundManager {
    tasks: Arc<Mutex<HashMap<String, TaskInfo>>>,
    notification_queue: Arc<Mutex<Vec<Notification>>>,
}

impl BackgroundManager {
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(Mutex::new(HashMap::new())),
            notification_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Start a background thread, return task_id immediately.
    pub fn run(&self, command: &str) -> String {
        let task_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        self.tasks.lock().unwrap().insert(
            task_id.clone(),
            TaskInfo {
                status: "running".to_string(),
                result: String::new(),
                command: command.to_string(),
            },
        );

        let queue = Arc::clone(&self.notification_queue);
        let tasks = Arc::clone(&self.tasks);
        let tid = task_id.clone();
        let cmd = command.to_string();

        thread::spawn(move || {
            let (status, output) = match Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .output()
            {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let combined = format!("{}{}", stdout, stderr).trim().to_string();
                    let result = if combined.is_empty() {
                        "(no output)".to_string()
                    } else {
                        truncate_str(&combined, 50000).to_string()
                    };
                    ("completed".to_string(), result)
                }
                Err(e) => ("error".to_string(), format!("Error: {}", e)),
            };

            // Update the task entry so check() reflects completion
            if let Ok(mut t) = tasks.lock() {
                if let Some(info) = t.get_mut(&tid) {
                    info.status = status.clone();
                    info.result = output.clone();
                }
            }

            let mut q = queue.lock().unwrap();
            q.push(Notification {
                task_id: tid,
                status,
                command: truncate_str(&cmd, 80).to_string(),
                result: truncate_str(&output, 500).to_string(),
            });
        });

        format!("Background task {} started: {}", task_id, truncate_str(command, 80))
    }

    /// Check status of one task.
    pub fn check(&self, task_id: &str) -> String {
        let tasks = self.tasks.lock().unwrap();
        match tasks.get(task_id) {
            Some(t) => format!(
                "[{}] {}\n{}",
                t.status,
                truncate_str(&t.command, 60),
                if t.result.is_empty() {
                    "(running)"
                } else {
                    &t.result
                }
            ),
            None => format!("Error: Unknown task {}", task_id),
        }
    }

    /// Return and clear all pending completion notifications.
    pub fn drain_notifications(&self) -> Vec<Notification> {
        let mut q = self.notification_queue.lock().unwrap();
        q.drain(..).collect()
    }
}


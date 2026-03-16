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

pub struct BackgroundManager {
    pub tasks: HashMap<String, TaskInfo>,
    notification_queue: Arc<Mutex<Vec<Notification>>>,
}

impl BackgroundManager {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            notification_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Start a background thread, return task_id immediately.
    pub fn run(&mut self, command: &str) -> String {
        let task_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        self.tasks.insert(
            task_id.clone(),
            TaskInfo {
                status: "running".to_string(),
                result: String::new(),
                command: command.to_string(),
            },
        );

        let queue = Arc::clone(&self.notification_queue);
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
                    } else if combined.len() > 50000 {
                        combined[..50000].to_string()
                    } else {
                        combined
                    };
                    ("completed".to_string(), result)
                }
                Err(e) => ("error".to_string(), format!("Error: {}", e)),
            };

            let mut q = queue.lock().unwrap();
            q.push(Notification {
                task_id: tid.clone(),
                status: status.clone(),
                command: if cmd.len() > 80 {
                    cmd[..80].to_string()
                } else {
                    cmd.clone()
                },
                result: if output.len() > 500 {
                    output[..500].to_string()
                } else {
                    output.clone()
                },
            });
        });

        format!("Background task {} started: {}", task_id, &command[..command.len().min(80)])
    }

    /// Check status of one task.
    pub fn check(&self, task_id: &str) -> String {
        match self.tasks.get(task_id) {
            Some(t) => format!(
                "[{}] {}\n{}",
                t.status,
                &t.command[..t.command.len().min(60)],
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
        let notifs: Vec<Notification> = q.drain(..).collect();
        notifs
    }
}

// Convert Notification to a HashMap-like structure for tests
impl Notification {
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("task_id".to_string(), self.task_id.clone());
        m.insert("status".to_string(), self.status.clone());
        m.insert("command".to_string(), self.command.clone());
        m.insert("result".to_string(), self.result.clone());
        m
    }
}

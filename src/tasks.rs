use crate::types::AgentError;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: i64,
    pub subject: String,
    #[serde(default)]
    pub description: String,
    pub status: String,
    #[serde(rename = "blockedBy", default)]
    pub blocked_by: Vec<i64>,
    #[serde(default)]
    pub blocks: Vec<i64>,
    #[serde(default)]
    pub owner: String,
    #[serde(default)]
    pub worktree: String,
}

pub struct TaskManager {
    pub dir: PathBuf,
    next_id: Cell<i64>,
}

impl TaskManager {
    pub fn new(dir: &Path) -> Self {
        std::fs::create_dir_all(dir).ok();
        let max_id = Self::max_id(dir);
        Self {
            dir: dir.to_path_buf(),
            next_id: Cell::new(max_id + 1),
        }
    }

    fn max_id(dir: &Path) -> i64 {
        let mut max = 0i64;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("task_") && name.ends_with(".json") {
                    if let Ok(id) = name
                        .trim_start_matches("task_")
                        .trim_end_matches(".json")
                        .parse::<i64>()
                    {
                        if id > max {
                            max = id;
                        }
                    }
                }
            }
        }
        max
    }

    fn task_path(&self, task_id: i64) -> PathBuf {
        self.dir.join(format!("task_{}.json", task_id))
    }

    fn load(&self, task_id: i64) -> Result<Task, AgentError> {
        let path = self.task_path(task_id);
        let content = std::fs::read_to_string(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                AgentError::NotFound(format!("Task {} not found", task_id))
            } else {
                AgentError::Io(e)
            }
        })?;
        let task: Task =
            serde_json::from_str(&content).map_err(|e| AgentError::ValueError(e.to_string()))?;
        Ok(task)
    }

    fn save(&self, task: &Task) -> Result<(), AgentError> {
        let path = self.task_path(task.id);
        let content =
            serde_json::to_string_pretty(task).map_err(|e| AgentError::ValueError(e.to_string()))?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    pub fn create(&self, subject: &str) -> String {
        let id = self.next_id.get();
        let task = Task {
            id,
            subject: subject.to_string(),
            description: String::new(),
            status: "pending".to_string(),
            blocked_by: Vec::new(),
            blocks: Vec::new(),
            owner: String::new(),
            worktree: String::new(),
        };
        self.save(&task).ok();
        self.next_id.set(id + 1);
        serde_json::to_string_pretty(&task).unwrap()
    }

    pub fn get(&self, task_id: i64) -> String {
        match self.load(task_id) {
            Ok(task) => serde_json::to_string_pretty(&task).unwrap(),
            Err(e) => format!("Error: {}", e),
        }
    }

    pub fn update(
        &self,
        task_id: i64,
        status: Option<&str>,
        add_blocked_by: Option<&[i64]>,
        add_blocks: Option<&[i64]>,
    ) -> Result<String, AgentError> {
        let mut task = self.load(task_id)?;

        if let Some(s) = status {
            if !["pending", "in_progress", "completed"].contains(&s) {
                return Err(AgentError::ValueError(format!("Invalid status: {}", s)));
            }
            task.status = s.to_string();
            if s == "completed" {
                self.clear_dependency(task_id)?;
            }
        }

        if let Some(blocked_by) = add_blocked_by {
            for &id in blocked_by {
                if !task.blocked_by.contains(&id) {
                    task.blocked_by.push(id);
                }
            }
        }

        if let Some(blocks) = add_blocks {
            for &blocked_id in blocks {
                if !task.blocks.contains(&blocked_id) {
                    task.blocks.push(blocked_id);
                }
                // Bidirectional: update the blocked task's blockedBy
                if let Ok(mut blocked_task) = self.load(blocked_id) {
                    if !blocked_task.blocked_by.contains(&task_id) {
                        blocked_task.blocked_by.push(task_id);
                        self.save(&blocked_task)?;
                    }
                }
            }
        }

        self.save(&task)?;
        Ok(serde_json::to_string_pretty(&task).unwrap())
    }

    /// Load all tasks from the directory.
    pub fn load_all(&self) -> Vec<Task> {
        load_all_tasks(&self.dir)
    }

    fn clear_dependency(&self, completed_id: i64) -> Result<(), AgentError> {
        for mut task in self.load_all() {
            if task.blocked_by.contains(&completed_id) {
                task.blocked_by.retain(|&id| id != completed_id);
                self.save(&task)?;
            }
        }
        Ok(())
    }

    pub fn list_all(&self) -> String {
        let mut tasks = self.load_all();
        tasks.sort_by_key(|t| t.id);
        if tasks.is_empty() {
            return "No tasks.".to_string();
        }
        tasks
            .iter()
            .map(|t| {
                let marker = match t.status.as_str() {
                    "pending" => "[ ]",
                    "in_progress" => "[>]",
                    "completed" => "[x]",
                    _ => "[?]",
                };
                let blocked = if !t.blocked_by.is_empty() {
                    format!(" (blocked by: {:?})", t.blocked_by)
                } else {
                    String::new()
                };
                format!("{} #{}: {}{}", marker, t.id, t.subject, blocked)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    // Methods for worktree integration (L11)
    pub fn exists(&self, task_id: i64) -> bool {
        self.task_path(task_id).exists()
    }

    pub fn bind_worktree(&self, task_id: i64, worktree: &str) -> Result<String, AgentError> {
        let mut task = self.load(task_id)?;
        task.worktree = worktree.to_string();
        if task.status == "pending" {
            task.status = "in_progress".to_string();
        }
        self.save(&task)?;
        Ok(serde_json::to_string_pretty(&task).unwrap())
    }

    pub fn unbind_worktree(&self, task_id: i64) -> Result<String, AgentError> {
        let mut task = self.load(task_id)?;
        task.worktree = String::new();
        self.save(&task)?;
        Ok(serde_json::to_string_pretty(&task).unwrap())
    }

    pub fn update_status(&self, task_id: i64, status: &str) -> Result<String, AgentError> {
        self.update(task_id, Some(status), None, None)
    }
}

/// Load all task JSON files from a directory. Shared by TaskManager and autonomy.
pub fn load_all_tasks(dir: &Path) -> Vec<Task> {
    let mut tasks = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("task_") && name_str.ends_with(".json") {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    if let Ok(task) = serde_json::from_str::<Task>(&content) {
                        tasks.push(task);
                    }
                }
            }
        }
    }
    tasks
}

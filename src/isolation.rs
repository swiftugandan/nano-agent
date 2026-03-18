use crate::tasks::TaskManager;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Event record for in-process subscribers (delegation, teammate, subagent)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EventRecord {
    pub event: String,
    pub ts: f64,
    pub data: serde_json::Value,
}

// ---------------------------------------------------------------------------
// EventBus: append-only lifecycle events (JSONL) + in-process subscribers
// ---------------------------------------------------------------------------

pub struct EventBus {
    pub log_path: PathBuf,
    subscribers: Mutex<Vec<EventSubscriber>>,
}

type EventSubscriber = Arc<dyn Fn(EventRecord) + Send + Sync>;

impl EventBus {
    pub fn new(log_path: &Path) -> Self {
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        Self {
            log_path: log_path.to_path_buf(),
            subscribers: Mutex::new(Vec::new()),
        }
    }

    /// Subscribe to all events (delegation, teammate, subagent, etc.).
    /// Callback is invoked from the thread that emits; safe to use from REPL or teammate threads.
    pub fn subscribe(&self, callback: EventSubscriber) {
        self.subscribers
            .lock()
            .expect("EventBus subscribers lock poisoned")
            .push(callback);
    }

    /// Emit an event with arbitrary JSON data. Appends to the log and notifies subscribers.
    /// Use for delegation_sent, teammate_started, subagent_progress, etc.
    pub fn emit_with_data(&self, event: &str, data: serde_json::Value) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let payload = serde_json::json!({
            "event": event,
            "ts": ts,
            "data": data,
        });
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.log_path)
                .unwrap();
            let line = serde_json::to_string(&payload).unwrap() + "\n";
            file.write_all(line.as_bytes()).unwrap();
        }
        let record = EventRecord {
            event: event.to_string(),
            ts,
            data,
        };
        let subs = self
            .subscribers
            .lock()
            .expect("EventBus subscribers lock poisoned");
        for cb in subs.iter() {
            cb(record.clone());
        }
    }

    pub fn emit(
        &self,
        event: &str,
        task: Option<serde_json::Value>,
        worktree: Option<serde_json::Value>,
        error: Option<&str>,
    ) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut payload = serde_json::json!({
            "event": event,
            "ts": timestamp,
            "task": task.unwrap_or(serde_json::json!({})),
            "worktree": worktree.unwrap_or(serde_json::json!({})),
        });

        if let Some(err) = error {
            payload["error"] = serde_json::Value::String(err.to_string());
        }

        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .unwrap();
        let line = serde_json::to_string(&payload).unwrap() + "\n";
        file.write_all(line.as_bytes()).unwrap();
    }

    pub fn list_recent(&self, limit: usize) -> String {
        let n = limit.clamp(1, 200);
        let content = std::fs::read_to_string(&self.log_path).unwrap_or_default();
        let lines: Vec<&str> = content.lines().collect();
        let start = if lines.len() > n { lines.len() - n } else { 0 };
        let recent: Vec<serde_json::Value> = lines[start..]
            .iter()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        serde_json::to_string_pretty(&recent).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// WorktreeManager
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeEntry {
    pub name: String,
    pub path: String,
    pub branch: String,
    #[serde(default)]
    pub task_id: Option<i64>,
    pub status: String,
    #[serde(default)]
    pub created_at: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub removed_at: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeIndex {
    pub worktrees: Vec<WorktreeEntry>,
}

pub struct WorktreeManager<'a> {
    pub repo_root: PathBuf,
    pub tasks: &'a TaskManager,
    pub events: &'a EventBus,
    pub dir: PathBuf,
    pub index_path: PathBuf,
}

impl<'a> WorktreeManager<'a> {
    pub fn new(repo_root: &Path, tasks: &'a TaskManager, events: &'a EventBus) -> Self {
        let dir = repo_root.join(".worktrees");
        std::fs::create_dir_all(&dir).ok();
        let index_path = dir.join("index.json");
        Self {
            repo_root: repo_root.to_path_buf(),
            tasks,
            events,
            dir,
            index_path,
        }
    }

    fn validate_name(&self, name: &str) -> Result<(), String> {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE.get_or_init(|| Regex::new(r"^[A-Za-z0-9._-]{1,40}$").unwrap());
        if !re.is_match(name) {
            return Err(
                "Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -".to_string(),
            );
        }
        Ok(())
    }

    fn load_index(&self) -> WorktreeIndex {
        let content = std::fs::read_to_string(&self.index_path).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or(WorktreeIndex {
            worktrees: Vec::new(),
        })
    }

    fn save_index(&self, index: &WorktreeIndex) {
        std::fs::write(
            &self.index_path,
            serde_json::to_string_pretty(index).unwrap(),
        )
        .ok();
    }

    fn find(&self, name: &str) -> Option<WorktreeEntry> {
        let index = self.load_index();
        index.worktrees.into_iter().find(|wt| wt.name == name)
    }

    fn run_git(&self, args: &[&str]) -> Result<String, String> {
        let result = Command::new("git")
            .args(args)
            .current_dir(&self.repo_root)
            .output()
            .map_err(|e| e.to_string())?;

        if !result.status.success() {
            let msg = format!(
                "{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            )
            .trim()
            .to_string();
            return Err(msg);
        }
        let output = format!(
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )
        .trim()
        .to_string();
        Ok(if output.is_empty() {
            "(no output)".to_string()
        } else {
            output
        })
    }

    pub fn create(&self, name: &str) -> Result<String, Box<dyn std::error::Error>> {
        self.create_with_task(name, None)
    }

    pub fn create_with_task(
        &self,
        name: &str,
        task_id: Option<i64>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.validate_name(name)
            .map_err(|e| -> Box<dyn std::error::Error> {
                Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
            })?;

        if self.find(name).is_some() {
            return Err(format!("Worktree '{}' already exists in index", name).into());
        }

        if let Some(tid) = task_id {
            if !self.tasks.exists(tid) {
                return Err(format!("Task {} not found", tid).into());
            }
        }

        let path = self.dir.join(name);
        let branch = format!("wt/{}", name);

        self.events.emit(
            "worktree.create.before",
            task_id.map(|id| serde_json::json!({"id": id})),
            Some(serde_json::json!({"name": name, "base_ref": "HEAD"})),
            None,
        );

        self.run_git(&[
            "worktree",
            "add",
            "-b",
            &branch,
            &path.to_string_lossy(),
            "HEAD",
        ])
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let entry = WorktreeEntry {
            name: name.to_string(),
            path: path.to_string_lossy().to_string(),
            branch: branch.clone(),
            task_id,
            status: "active".to_string(),
            created_at: timestamp,
            removed_at: None,
        };

        let mut index = self.load_index();
        index.worktrees.push(entry.clone());
        self.save_index(&index);

        if let Some(tid) = task_id {
            self.tasks.bind_worktree(tid, name).ok();
        }

        self.events.emit(
            "worktree.create.after",
            task_id.map(|id| serde_json::json!({"id": id})),
            Some(serde_json::json!({
                "name": name,
                "path": path.to_string_lossy(),
                "branch": branch,
                "status": "active",
            })),
            None,
        );

        Ok(serde_json::to_string_pretty(&entry).unwrap())
    }

    pub fn remove(&self, name: &str) -> Result<String, Box<dyn std::error::Error>> {
        self.remove_with_options(name, false)
    }

    pub fn remove_with_options(
        &self,
        name: &str,
        complete_task: bool,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let wt = self
            .find(name)
            .ok_or_else(|| format!("Unknown worktree '{}'", name))?;

        self.events.emit(
            "worktree.remove.before",
            wt.task_id.map(|id| serde_json::json!({"id": id})),
            Some(serde_json::json!({"name": name, "path": wt.path})),
            None,
        );

        self.run_git(&["worktree", "remove", &wt.path])
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

        if complete_task {
            if let Some(tid) = wt.task_id {
                self.tasks.update_status(tid, "completed").ok();
                self.tasks.unbind_worktree(tid).ok();
            }
        }

        // Update index
        let mut index = self.load_index();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        for item in &mut index.worktrees {
            if item.name == name {
                item.status = "removed".to_string();
                item.removed_at = Some(timestamp);
            }
        }
        self.save_index(&index);

        self.events.emit(
            "worktree.remove.after",
            wt.task_id.map(|id| serde_json::json!({"id": id})),
            Some(serde_json::json!({"name": name, "path": wt.path, "status": "removed"})),
            None,
        );

        Ok(format!("Removed worktree '{}'", name))
    }
}

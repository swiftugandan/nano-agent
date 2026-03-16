use crate::tasks::Task;
use std::path::Path;

/// Scan for unclaimed tasks: pending, no owner, empty blockedBy.
pub fn scan_unclaimed_tasks(dir: &Path) -> Vec<Task> {
    let mut unclaimed: Vec<Task> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.starts_with("task_") && n.ends_with(".json"))
            {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(task) = serde_json::from_str::<Task>(&content) {
                        if task.status == "pending"
                            && task.owner.is_empty()
                            && task.blocked_by.is_empty()
                        {
                            unclaimed.push(task);
                        }
                    }
                }
            }
        }
    }
    unclaimed
}

/// Claim a task: set owner and status to in_progress.
pub fn claim_task(dir: &Path, task_id: i64, owner: &str) -> String {
    let path = dir.join(format!("task_{}.json", task_id));
    if !path.exists() {
        return format!("Error: Task {} not found", task_id);
    }
    match std::fs::read_to_string(&path) {
        Ok(content) => match serde_json::from_str::<Task>(&content) {
            Ok(mut task) => {
                task.owner = owner.to_string();
                task.status = "in_progress".to_string();
                std::fs::write(&path, serde_json::to_string_pretty(&task).unwrap()).ok();
                format!("Claimed task #{} for {}", task_id, owner)
            }
            Err(e) => format!("Error: {}", e),
        },
        Err(e) => format!("Error: {}", e),
    }
}

/// Create an identity block for injection after compression.
pub fn make_identity_block(name: &str, role: &str, team: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "user",
        "content": format!("<identity>You are '{}', role: {}, team: {}. Continue your work.</identity>", name, role, team),
    })
}

/// Should inject identity? Returns true if messages.len() <= 3.
pub fn should_inject(messages: &[serde_json::Value]) -> bool {
    messages.len() <= 3
}

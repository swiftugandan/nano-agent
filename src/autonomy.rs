use crate::tasks::{load_all_tasks, Task};
use std::path::Path;

/// Scan for unclaimed tasks: pending, no owner, empty blockedBy.
pub fn scan_unclaimed_tasks(dir: &Path) -> Vec<Task> {
    let mut unclaimed: Vec<Task> = load_all_tasks(dir)
        .into_iter()
        .filter(|task| {
            task.status == "pending" && task.owner.is_empty() && task.blocked_by.is_empty()
        })
        .collect();
    unclaimed.sort_by_key(|t| t.id);
    unclaimed
}

/// Claim a task: set owner and status to in_progress.
pub fn claim_task(dir: &Path, task_id: i64, owner: &str) -> String {
    let path = dir.join(format!("task_{}.json", task_id));
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

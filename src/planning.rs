use crate::types::AgentError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TodoItem + TodoManager
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: String,
    pub text: String,
    pub status: String,
}

pub struct TodoManager {
    pub items: Vec<TodoItem>,
}

impl TodoManager {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Update the todo list. Validates constraints:
    /// - max 1 in_progress
    /// - non-empty text
    /// - valid status values
    pub fn update(&mut self, items: Vec<TodoItem>) -> Result<String, AgentError> {
        let mut in_progress_count = 0;
        let valid_statuses = ["pending", "in_progress", "completed"];

        for item in &items {
            if item.text.trim().is_empty() {
                return Err(AgentError::ValueError(format!(
                    "Item {}: text required",
                    item.id
                )));
            }
            if !valid_statuses.contains(&item.status.as_str()) {
                return Err(AgentError::ValueError(format!(
                    "Item {}: invalid status '{}'",
                    item.id, item.status
                )));
            }
            if item.status == "in_progress" {
                in_progress_count += 1;
            }
        }
        if in_progress_count > 1 {
            return Err(AgentError::ValueError(
                "Only one task can be in_progress at a time".to_string(),
            ));
        }

        self.items = items;
        Ok(self.render())
    }

    /// Render the todo list as a string.
    pub fn render(&self) -> String {
        if self.items.is_empty() {
            return "No todos.".to_string();
        }
        let mut lines: Vec<String> = self
            .items
            .iter()
            .map(|item| {
                let marker = match item.status.as_str() {
                    "pending" => "[ ]",
                    "in_progress" => "[>]",
                    "completed" => "[x]",
                    _ => "[?]",
                };
                format!("{} #{}: {}", marker, item.id, item.text)
            })
            .collect();
        let done = self.items.iter().filter(|i| i.status == "completed").count();
        lines.push(format!("\n({}/{} completed)", done, self.items.len()));
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// NagPolicy
// ---------------------------------------------------------------------------

pub struct NagPolicy {
    pub rounds_since_todo: usize,
    pub threshold: usize,
}

impl NagPolicy {
    pub fn new(threshold: usize) -> Self {
        Self {
            rounds_since_todo: 0,
            threshold,
        }
    }

    pub fn tick(&mut self) {
        self.rounds_since_todo += 1;
    }

    pub fn should_inject(&self) -> bool {
        self.rounds_since_todo >= self.threshold
    }

    pub fn reset(&mut self) {
        self.rounds_since_todo = 0;
    }
}

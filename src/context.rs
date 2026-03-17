use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::knowledge::SkillLoader;
use crate::memory_store::MemoryStore;
use crate::planning::TodoManager;
use crate::prompt::format_recalled_memories;
use crate::tasks::TaskManager;

// ---------------------------------------------------------------------------
// Projector — demand-paging write gate for context entries
// ---------------------------------------------------------------------------

pub struct Projector {
    spill_dir: PathBuf,
    threshold: usize, // chars; content below this passes through unchanged
}

impl Projector {
    pub fn new(spill_dir: &Path, threshold: usize) -> Self {
        Self {
            spill_dir: spill_dir.to_path_buf(),
            threshold,
        }
    }

    /// Project content for context-efficient inclusion.
    /// Below threshold: returns content unchanged.
    /// Above threshold: spills to disk, returns head + breadcrumb.
    pub fn project(&self, namespace: &str, key: &str, content: &str) -> String {
        if content.len() <= self.threshold {
            return content.to_string();
        }
        let ns_dir = self.spill_dir.join(namespace);
        let written = fs::create_dir_all(&ns_dir)
            .and_then(|_| fs::write(ns_dir.join(key), content))
            .is_ok();
        if !written {
            // Disk write failed — return full content rather than losing it
            return content.to_string();
        }
        let head = truncate_at_boundary(content, self.threshold);
        format!(
            "{}\n\n[… {} bytes total. Retrieve with projector://{}/{}]",
            head,
            content.len(),
            namespace,
            key
        )
    }

    /// Retrieve spilled content by namespace/key.
    pub fn retrieve(&self, namespace: &str, key: &str) -> Option<String> {
        fs::read_to_string(self.spill_dir.join(namespace).join(key)).ok()
    }
}

/// Truncate content at a word/line boundary near `max_chars`.
/// Uses char count (not byte length) and snaps to a valid UTF-8 char boundary.
fn truncate_at_boundary(content: &str, max_chars: usize) -> &str {
    // Find the byte offset of the max_chars-th character
    let byte_limit = content
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(content.len());
    if byte_limit >= content.len() {
        return content;
    }
    let window = &content[..byte_limit];
    // Try to break at a newline within the last 20% of the window
    let search_start = byte_limit.saturating_sub(byte_limit / 5);
    // Snap search_start to a char boundary
    let search_start = content[..search_start]
        .char_indices()
        .next_back()
        .map(|(i, _)| i)
        .unwrap_or(0);
    if let Some(pos) = window[search_start..].rfind('\n') {
        return &content[..search_start + pos];
    }
    // Fall back to a space boundary
    if let Some(pos) = window[search_start..].rfind(' ') {
        return &content[..search_start + pos];
    }
    window
}

// ---------------------------------------------------------------------------
// Seed trait — uniform system prompt contribution
// ---------------------------------------------------------------------------

pub trait Seed: Send + Sync {
    /// Section name (becomes ## heading in system prompt)
    fn name(&self) -> &str;

    /// Compact representation for the system prompt.
    /// Empty string = omit this section.
    fn seed(&self) -> String;

    /// On-demand retrieval of full content for a specific item.
    fn retrieve(&self, _key: &str) -> Option<String> {
        None
    }
}

// ---------------------------------------------------------------------------
// SeedCollector — gathers seeds for prompt assembly
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct SeedCollector {
    seeds: Vec<Arc<dyn Seed>>,
}

impl SeedCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, seed: Arc<dyn Seed>) {
        self.seeds.push(seed);
    }

    /// Render all seeds as (name, text) pairs for the system prompt.
    pub fn render(&self) -> Vec<(String, String)> {
        self.seeds
            .iter()
            .map(|s| (s.name().to_string(), s.seed()))
            .filter(|(_, text)| !text.is_empty())
            .collect()
    }

    /// Route a retrieval call to the right seed by name.
    pub fn retrieve(&self, name: &str, key: &str) -> Option<String> {
        self.seeds
            .iter()
            .find(|s| s.name() == name)
            .and_then(|s| s.retrieve(key))
    }
}

// ---------------------------------------------------------------------------
// Seed implementations — thin wrappers around existing subsystems
// ---------------------------------------------------------------------------

/// Seed for skill descriptions (compact list) with on-demand full content.
pub struct SkillSeed {
    loader: Arc<SkillLoader>,
}

impl SkillSeed {
    pub fn new(loader: Arc<SkillLoader>) -> Self {
        Self { loader }
    }
}

impl Seed for SkillSeed {
    fn name(&self) -> &str {
        "Available Skills"
    }

    fn seed(&self) -> String {
        self.loader.get_descriptions()
    }

    fn retrieve(&self, key: &str) -> Option<String> {
        let content = self.loader.get_content(key);
        if content.starts_with("Error: ") {
            None
        } else {
            Some(content)
        }
    }
}

/// Seed for the current todo list.
pub struct TodoSeed {
    manager: Arc<RwLock<TodoManager>>,
}

impl TodoSeed {
    pub fn new(manager: Arc<RwLock<TodoManager>>) -> Self {
        Self { manager }
    }
}

impl Seed for TodoSeed {
    fn name(&self) -> &str {
        "Current Todo List"
    }

    fn seed(&self) -> String {
        self.manager
            .read()
            .expect("TodoManager read lock poisoned")
            .render()
    }
}

/// Seed for recalled memories (TF-IDF search against a query).
/// Caches last query+result to avoid redundant TF-IDF recall.
pub struct MemorySeed {
    store: Arc<MemoryStore>,
    query: std::sync::RwLock<String>,
    cached: std::sync::RwLock<(String, String)>, // (last_query, last_result)
}

impl MemorySeed {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self {
            store,
            query: std::sync::RwLock::new(String::new()),
            cached: std::sync::RwLock::new((String::new(), String::new())),
        }
    }

    /// Update the query used for recall (called each turn with user input).
    pub fn set_query(&self, query: &str) {
        *self.query.write().expect("MemorySeed query lock poisoned") = query.to_string();
    }
}

impl Seed for MemorySeed {
    fn name(&self) -> &str {
        "Recalled Memories"
    }

    fn seed(&self) -> String {
        let query = self.query.read().expect("MemorySeed query lock poisoned");
        if query.is_empty() {
            return String::new();
        }
        // Return cached result if query unchanged
        {
            let cached = self.cached.read().expect("MemorySeed cache lock poisoned");
            if cached.0 == *query {
                return cached.1.clone();
            }
        }
        let entries = self.store.recall(&query, 3);
        let result = format_recalled_memories(&entries);
        // Cache for next call
        *self.cached.write().expect("MemorySeed cache lock poisoned") =
            (query.clone(), result.clone());
        result
    }
}

/// Seed for the task list overview with on-demand task detail.
pub struct TaskSeed {
    manager: Arc<RwLock<TaskManager>>,
}

impl TaskSeed {
    pub fn new(manager: Arc<RwLock<TaskManager>>) -> Self {
        Self { manager }
    }
}

impl Seed for TaskSeed {
    fn name(&self) -> &str {
        "Tasks"
    }

    fn seed(&self) -> String {
        let mgr = self.manager.read().expect("TaskManager read lock poisoned");
        let listing = mgr.list_all();
        if listing.contains("No tasks") {
            return String::new();
        }
        listing
    }

    fn retrieve(&self, key: &str) -> Option<String> {
        let task_id: i64 = key.parse().ok()?;
        let mgr = self.manager.read().expect("TaskManager read lock poisoned");
        let result = mgr.get(task_id);
        if result.starts_with("Error: ") {
            None
        } else {
            Some(result)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn projector_passthrough_below_threshold() {
        let tmp = TempDir::new().unwrap();
        let proj = Projector::new(tmp.path(), 100);
        let content = "short content";
        assert_eq!(proj.project("ns", "k", content), content);
    }

    #[test]
    fn projector_spills_above_threshold() {
        let tmp = TempDir::new().unwrap();
        let proj = Projector::new(tmp.path(), 20);
        let content =
            "This is a much longer string that exceeds the threshold limit by quite a bit";
        let result = proj.project("ns", "k", content);
        assert!(result.contains("bytes total. Retrieve with projector://"));
        // Verify spill file exists
        let spilled = fs::read_to_string(tmp.path().join("ns").join("k")).unwrap();
        assert_eq!(spilled, content);
    }

    #[test]
    fn projector_retrieve_returns_spilled_content() {
        let tmp = TempDir::new().unwrap();
        let proj = Projector::new(tmp.path(), 20);
        let content =
            "This is a much longer string that exceeds the threshold limit by quite a bit";
        proj.project("tool_results", "abc123", content);
        let retrieved = proj.retrieve("tool_results", "abc123").unwrap();
        assert_eq!(retrieved, content);
    }

    #[test]
    fn projector_retrieve_missing_returns_none() {
        let tmp = TempDir::new().unwrap();
        let proj = Projector::new(tmp.path(), 100);
        assert!(proj.retrieve("ns", "missing").is_none());
    }

    #[test]
    fn truncate_at_newline_boundary() {
        let content = "line one\nline two\nline three\nline four\nline five";
        let result = truncate_at_boundary(content, 30);
        assert!(result.len() <= 30);
        assert!(result.ends_with('\n') || !result.contains("line four"));
    }

    #[test]
    fn seed_collector_filters_empty() {
        struct EmptySeed;
        impl Seed for EmptySeed {
            fn name(&self) -> &str {
                "Empty"
            }
            fn seed(&self) -> String {
                String::new()
            }
        }
        struct NonEmptySeed;
        impl Seed for NonEmptySeed {
            fn name(&self) -> &str {
                "NonEmpty"
            }
            fn seed(&self) -> String {
                "content".into()
            }
        }

        let mut collector = SeedCollector::new();
        collector.register(Arc::new(EmptySeed));
        collector.register(Arc::new(NonEmptySeed));
        let rendered = collector.render();
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].0, "NonEmpty");
    }
}

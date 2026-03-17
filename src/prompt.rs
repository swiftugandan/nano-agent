use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Prompt context passed to the assembler
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct PromptContext {
    pub agent_name: String,
    pub agent_role: String,
    pub cwd: String,
    pub tool_count: usize,
    pub todo_state: String,
    pub skill_descriptions: String,
    // GAP 4: Runtime context injection
    pub timestamp: String,
    pub model_id: String,
    pub agent_id: String,
    pub session_id: String,
    // GAP 3/5: Recalled memories from TF-IDF search
    pub recalled_memories: String,
}

// ---------------------------------------------------------------------------
// PromptAssembler: composes the system prompt from layered files
// ---------------------------------------------------------------------------

pub struct PromptAssembler {
    prompts_dir: PathBuf,
    layers: Vec<(String, String)>, // (filename, content)
}

// GAP 5: Required vs optional layer definitions
const REQUIRED_LAYERS: &[&str] = &[
    "SOUL.md",
    "IDENTITY.md",
    "TOOLS.md",
    "GUIDELINES.md",
    "MEMORY.md",
];

const OPTIONAL_LAYERS: &[&str] = &["HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "USER.md"];

impl PromptAssembler {
    pub fn new(prompts_dir: &Path) -> Self {
        let mut assembler = Self {
            prompts_dir: prompts_dir.to_path_buf(),
            layers: Vec::new(),
        };
        assembler.load();
        assembler
    }

    /// Load all layer files from the prompts directory.
    /// Required layers are loaded first, then optional layers (silently skipped if absent).
    fn load(&mut self) {
        self.layers.clear();
        for filename in REQUIRED_LAYERS.iter().chain(OPTIONAL_LAYERS.iter()) {
            let path = self.prompts_dir.join(filename);
            if let Ok(content) = fs::read_to_string(&path) {
                if !content.trim().is_empty() {
                    self.layers.push((filename.to_string(), content));
                }
            }
        }
    }

    /// Create default prompt files if the prompts directory is empty or missing.
    pub fn init_defaults(&self) {
        fs::create_dir_all(&self.prompts_dir).ok();

        let defaults = vec![
            (
                "SOUL.md",
                "You are an autonomous coding agent named '{name}' (role: {role}) working in {cwd}.\n\
                 You operate independently, making decisions and using tools to accomplish tasks.\n",
            ),
            (
                "IDENTITY.md",
                "## Identity\n\
                 - Agent: {name}\n\
                 - Role: {role}\n\
                 - Model: {model_id}\n\
                 - Agent ID: {agent_id}\n\
                 - Session: {session_id}\n\
                 - Timestamp: {timestamp}\n",
            ),
            (
                "TOOLS.md",
                "## Your Capabilities\n\
                 You have {tool_count} tools spanning file I/O, task management, planning, background \
                 execution, team communication, git worktree isolation, and subagent delegation.\n",
            ),
            (
                "GUIDELINES.md",
                "## Guidelines\n\
                 - Use todo_update to track your work. Keep exactly one item in_progress.\n\
                 - Use task_create/task_update for persistent multi-step work with dependencies.\n\
                 - Use background_run for long-running commands so you don't block.\n\
                 - Use subagent to delegate isolated subtasks.\n\
                 - Use worktree_create to work in isolated git branches per task.\n\
                 - Communicate with teammates via send_message/broadcast_message.\n\
                 - Use scan_tasks and claim_task to pick up unclaimed work.\n\
                 - Use save_memory to persist important context for future recall.\n",
            ),
            (
                "HEARTBEAT.md",
                "## Heartbeat\n\
                 Cron jobs can be scheduled with cron_add to inject prompts on a schedule.\n\
                 Use cron_list to see active schedules.\n",
            ),
            (
                "BOOTSTRAP.md",
                "## Bootstrap\n\
                 On startup, review your todo list and any pending tasks.\n\
                 Check your inbox for messages from teammates.\n",
            ),
            (
                "AGENTS.md",
                "## Multi-Agent\n\
                 You can spawn teammates with spawn_teammate for parallel work.\n\
                 Teammates communicate via the message bus.\n",
            ),
            (
                "USER.md",
                "## User Context\n\
                 Adapt your responses to the user's expertise and preferences.\n",
            ),
        ];

        for (filename, content) in defaults {
            let path = self.prompts_dir.join(filename);
            if !path.exists() {
                fs::write(&path, content).ok();
            }
        }
    }

    /// Compose the full system prompt with placeholder substitution.
    pub fn compose(&self, ctx: &PromptContext) -> String {
        let mut parts: Vec<String> = Vec::new();

        // Static layers from files (with placeholder substitution)
        for (filename, content) in &self.layers {
            let substituted = substitute(content, ctx);
            // Insert dynamic sections after TOOLS.md
            parts.push(substituted);
            if filename == "TOOLS.md" {
                // Dynamic: todo state
                if !ctx.todo_state.is_empty() {
                    parts.push(format!("## Current Todo List\n{}", ctx.todo_state));
                }
                // Dynamic: skill descriptions
                if !ctx.skill_descriptions.is_empty() {
                    parts.push(format!("## Available Skills\n{}", ctx.skill_descriptions));
                }
            }
        }

        // GAP 5: Inject recalled memories if present
        if !ctx.recalled_memories.is_empty() {
            parts.push(format!("## Recalled Memories\n{}", ctx.recalled_memories));
        }

        // If no files were loaded, fall back to a minimal prompt
        if parts.is_empty() {
            return format!(
                "You are an autonomous coding agent named '{}' (role: {}) working in {}.\n\
                 You have {} tools available.",
                ctx.agent_name, ctx.agent_role, ctx.cwd, ctx.tool_count,
            );
        }

        parts.join("\n\n")
    }

    /// Reload prompt files from disk (useful if user edits them).
    pub fn reload(&mut self) {
        self.load();
    }
}

/// Replace placeholders in a template string.
fn substitute(template: &str, ctx: &PromptContext) -> String {
    template
        .replace("{name}", &ctx.agent_name)
        .replace("{role}", &ctx.agent_role)
        .replace("{cwd}", &ctx.cwd)
        .replace("{tool_count}", &ctx.tool_count.to_string())
        .replace("{timestamp}", &ctx.timestamp)
        .replace("{model_id}", &ctx.model_id)
        .replace("{agent_id}", &ctx.agent_id)
        .replace("{session_id}", &ctx.session_id)
}

// ---------------------------------------------------------------------------
// Prompt helpers (moved from main.rs)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn build_prompt_context(
    agent_name: &str,
    agent_role: &str,
    cwd: &Path,
    tool_count: usize,
    todo_state: String,
    skill_desc: String,
    model_name: &str,
    agent_id: &str,
    session_id: &str,
    recalled_memories: String,
) -> PromptContext {
    PromptContext {
        agent_name: agent_name.to_string(),
        agent_role: agent_role.to_string(),
        cwd: cwd.display().to_string(),
        tool_count,
        todo_state: if todo_state.is_empty() {
            "(empty)".into()
        } else {
            todo_state
        },
        skill_descriptions: if skill_desc.is_empty() {
            "(none loaded)".into()
        } else {
            skill_desc
        },
        timestamp: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
        model_id: model_name.to_string(),
        agent_id: agent_id.to_string(),
        session_id: session_id.to_string(),
        recalled_memories,
    }
}

pub fn format_recalled_memories(entries: &[crate::memory_store::MemoryEntry]) -> String {
    if entries.is_empty() {
        String::new()
    } else {
        entries
            .iter()
            .map(|m| format!("- [{}] {}", m.timestamp, m.text))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub fn extract_last_response_text(messages: &[serde_json::Value]) -> Option<String> {
    let last = messages.last()?;
    let content = last.get("content")?;
    if let Some(arr) = content.as_array() {
        let texts: Vec<&str> = arr
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join(""))
        }
    } else {
        content.as_str().map(|s| s.to_string())
    }
}

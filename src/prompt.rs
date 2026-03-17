use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Prompt context passed to the assembler
// ---------------------------------------------------------------------------

pub struct PromptContext {
    pub agent_name: String,
    pub agent_role: String,
    pub cwd: String,
    pub tool_count: usize,
    pub todo_state: String,
    pub skill_descriptions: String,
}

// ---------------------------------------------------------------------------
// PromptAssembler: composes the system prompt from layered files
// ---------------------------------------------------------------------------

pub struct PromptAssembler {
    prompts_dir: PathBuf,
    layers: Vec<(String, String)>, // (filename, content)
}

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
    fn load(&mut self) {
        self.layers.clear();
        let layer_files = [
            "SOUL.md",
            "IDENTITY.md",
            "TOOLS.md",
            "GUIDELINES.md",
            "MEMORY.md",
        ];
        for filename in &layer_files {
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
                 - Role: {role}\n",
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
                 - Use scan_tasks and claim_task to pick up unclaimed work.\n",
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
}

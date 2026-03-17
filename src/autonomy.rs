use crate::core_loop::run_agent_loop;
use crate::tasks::{load_all_tasks, Task};
use crate::teams::{MessageBus, TeammateManager};
use crate::types::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

/// Format inbox messages for injection into the conversation.
pub fn format_inbox(inbox: &[serde_json::Value]) -> String {
    format!(
        "[Inbox: {} message(s)]\n{}",
        inbox.len(),
        serde_json::to_string_pretty(inbox).unwrap_or_default()
    )
}

// ---------------------------------------------------------------------------
// Work/Idle Lifecycle (Gap 2)
// ---------------------------------------------------------------------------

pub struct LifecycleConfig {
    pub poll_interval: Duration,
    pub idle_timeout: Duration,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(60),
        }
    }
}

/// Bundles the shared dependencies for a teammate lifecycle.
pub struct LifecycleContext {
    pub teammate_manager: Arc<Mutex<TeammateManager>>,
    pub message_bus: Arc<MessageBus>,
    pub tasks_dir: PathBuf,
    pub transcript_dir: PathBuf,
    pub agent_name: String,
    pub idle_signal: Arc<AtomicBool>,
}

/// Run a teammate through a WORK → IDLE → SHUTDOWN lifecycle.
///
/// - WORK phase: runs `run_agent_loop` until the agent calls the `idle` tool
///   (which sets `idle_signal`), then transitions to IDLE.
/// - IDLE phase: polls for shutdown requests, inbox messages, or unclaimed tasks.
///   If a message or task arrives, transitions back to WORK.
///   If `idle_timeout` elapses with no activity, auto-shuts down.
/// - SHUTDOWN: sets status to "shutdown" and returns.
pub fn run_teammate_lifecycle(
    llm: &mut dyn Llm,
    system: &str,
    messages: &mut Vec<serde_json::Value>,
    tools: &[serde_json::Value],
    dispatch: &Dispatch,
    ctx: &LifecycleContext,
    config: &LifecycleConfig,
) {
    let idle_signal = &ctx.idle_signal;
    loop {
        // --- WORK phase ---
        idle_signal.store(false, Ordering::Release);
        ctx.teammate_manager
            .lock()
            .unwrap()
            .set_status(&ctx.agent_name, "working");

        let signals = LoopSignals {
            compact_signal: None,
            transcript_dir: Some(&ctx.transcript_dir),
            idle_signal: Some(idle_signal),
        };

        run_agent_loop(llm, system, messages, tools, dispatch, &signals);

        // If idle_signal wasn't set (i.e. LLM stopped for another reason), shut down
        if !idle_signal.load(Ordering::Acquire) {
            ctx.teammate_manager
                .lock()
                .unwrap()
                .set_status(&ctx.agent_name, "shutdown");
            return;
        }

        // --- IDLE phase ---
        idle_signal.store(false, Ordering::Release);
        ctx.teammate_manager
            .lock()
            .unwrap()
            .set_status(&ctx.agent_name, "idle");

        let idle_start = std::time::Instant::now();

        loop {
            // Check shutdown
            if ctx
                .teammate_manager
                .lock()
                .unwrap()
                .is_shutdown_requested(&ctx.agent_name)
            {
                return; // already marked shutdown
            }

            // Check inbox
            let inbox = ctx.message_bus.read_inbox(&ctx.agent_name);
            if !inbox.is_empty() {
                messages.push(serde_json::json!({"role": "user", "content": format_inbox(&inbox)}));
                break; // back to WORK
            }

            // Check unclaimed tasks
            let unclaimed = scan_unclaimed_tasks(&ctx.tasks_dir);
            if !unclaimed.is_empty() {
                let text = format!(
                    "[System: {} unclaimed task(s) available. Use scan_tasks + claim_task to pick one up.]",
                    unclaimed.len()
                );
                messages.push(serde_json::json!({"role": "user", "content": text}));
                break; // back to WORK
            }

            // Check timeout
            if idle_start.elapsed() >= config.idle_timeout {
                ctx.teammate_manager
                    .lock()
                    .unwrap()
                    .set_status(&ctx.agent_name, "shutdown");
                return;
            }

            std::thread::sleep(config.poll_interval);
        }
    }
}

use crate::core_loop::run_agent_loop;
use crate::handler::{AgentContext, HandlerRegistry};
use crate::tasks::{load_all_tasks, Task};
use crate::types::*;
use std::path::Path;
use std::sync::atomic::Ordering;
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

/// Create an assistant acknowledgment message for identity injection.
pub fn make_acknowledgment(name: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": format!("I am {}. Continuing.", name),
    })
}

/// Should inject identity? Returns true if messages.len() <= 3.
pub fn should_inject(messages: &[serde_json::Value]) -> bool {
    messages.len() <= 3
}

/// Inject identity block + acknowledgment into messages if the conversation is fresh.
/// Inserts identity at position 0 and acknowledgment at position 1.
pub fn inject_identity_if_needed(
    messages: &mut Vec<serde_json::Value>,
    name: &str,
    role: &str,
    team: &str,
) {
    if should_inject(messages) {
        messages.insert(0, make_identity_block(name, role, team));
        messages.insert(1, make_acknowledgment(name));
    }
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
// IdlePolicy: reusable idle-phase decision engine (L10)
// ---------------------------------------------------------------------------

/// Result of a single idle poll cycle.
#[derive(Debug, Clone, PartialEq)]
pub enum PollResult {
    /// Unclaimed task found — claim and resume.
    Claim,
    /// Nothing to do yet — keep waiting.
    Wait,
    /// Idle timeout exceeded — shut down.
    Shutdown,
}

/// Encapsulates idle-phase polling logic as a reusable interface.
pub struct IdlePolicy {
    pub poll_interval: Duration,
    pub idle_timeout: Duration,
}

impl IdlePolicy {
    pub fn new(poll_interval: Duration, idle_timeout: Duration) -> Self {
        Self {
            poll_interval,
            idle_timeout,
        }
    }

    /// Evaluate the current idle state and return the appropriate action.
    /// Note: inbox checking is handled by the caller (since read_inbox is destructive).
    pub fn poll(&self, unclaimed: &[Task], elapsed: Duration) -> PollResult {
        if !unclaimed.is_empty() {
            return PollResult::Claim;
        }
        if elapsed >= self.idle_timeout {
            return PollResult::Shutdown;
        }
        PollResult::Wait
    }
}

impl Default for IdlePolicy {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(60),
        }
    }
}

// ---------------------------------------------------------------------------
// Work/Idle Lifecycle
// ---------------------------------------------------------------------------

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
    registry: &HandlerRegistry,
    ctx: &AgentContext,
    policy: &IdlePolicy,
) {
    let idle_signal = &ctx.signals.idle;

    loop {
        // --- WORK phase ---
        idle_signal.store(false, Ordering::Release);
        ctx.services
            .teammate_manager
            .write()
            .expect("TeammateManager write lock poisoned")
            .set_status(&ctx.identity.name, "working");
        ctx.services.event_bus.emit_with_data(
            "teammate_started",
            serde_json::json!({ "name": ctx.identity.name, "role": ctx.identity.role }),
        );

        run_agent_loop(llm, system, messages, tools, registry, ctx);

        // If idle_signal wasn't set, the LLM ended without calling the idle tool (e.g. first
        // turn it just said "I'm ready" and stopped). Treat that as "no work yet" and go to IDLE
        // so the teammate can poll for inbox/tasks instead of shutting down immediately.
        if !idle_signal.load(Ordering::Acquire) {
            // Fall through to IDLE phase instead of return
        }

        // --- IDLE phase ---
        idle_signal.store(false, Ordering::Release);
        ctx.services
            .teammate_manager
            .write()
            .expect("TeammateManager write lock poisoned")
            .set_status(&ctx.identity.name, "idle");

        let idle_start = std::time::Instant::now();

        loop {
            // Check shutdown request (external)
            if ctx
                .services
                .teammate_manager
                .read()
                .expect("TeammateManager read lock poisoned")
                .is_shutdown_requested(&ctx.identity.name)
            {
                return;
            }

            // Check inbox first — read_inbox is destructive (drains file),
            // so we must consume the result immediately if non-empty.
            let inbox = ctx.services.message_bus.read_inbox(&ctx.identity.name);
            if !inbox.is_empty() {
                ctx.services.event_bus.emit_with_data(
                    "teammate_received",
                    serde_json::json!({
                        "name": ctx.identity.name,
                        "message_count": inbox.len(),
                        "from": inbox
                            .first()
                            .and_then(|m| m.get("from"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("?"),
                    }),
                );
                messages.push(serde_json::json!({"role": "user", "content": format_inbox(&inbox)}));
                break; // back to WORK
            }

            // Inbox was empty (drain was a no-op). Now check tasks + timeout.
            let unclaimed = scan_unclaimed_tasks(&ctx.tasks_dir);
            match policy.poll(&unclaimed, idle_start.elapsed()) {
                PollResult::Claim => {
                    let text = format!(
                        "[System: {} unclaimed task(s) available. Use scan_tasks + claim_task to pick one up.]",
                        unclaimed.len()
                    );
                    messages.push(serde_json::json!({"role": "user", "content": text}));
                    break; // back to WORK
                }
                PollResult::Shutdown => {
                    ctx.services
                        .teammate_manager
                        .write()
                        .expect("TeammateManager write lock poisoned")
                        .set_status(&ctx.identity.name, "shutdown");
                    return;
                }
                PollResult::Wait => {
                    std::thread::sleep(policy.poll_interval);
                }
            }
        }
    }
}

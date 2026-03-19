use crate::autonomy;
use crate::context::MemorySeed;
use crate::context::Projector;
use crate::handler::AgentContext;
use crate::memory;
use crate::types::Llm;

/// UI hooks for the pre-turn pipeline (TUI/REPL/headless).
pub trait PreTurnUi: Send + Sync {
    fn compact_notice(&self, path: &str);
    fn background_notification(&self, source: &str, message: &str);
}

/// Runs identity injection, token compaction, nag policy, background notifications,
/// inbox drain, and heartbeat drain. Returns the token count so the caller can reuse
/// it for prompt-state updates.
#[allow(clippy::too_many_arguments)]
pub fn build_pre_turn_context(
    messages: &mut Vec<serde_json::Value>,
    ctx: &AgentContext,
    llm: &mut dyn Llm,
    transcript_dir: &std::path::Path,
    projector: &Projector,
    turn_count: usize,
    memory_seed: &MemorySeed,
    input: &str,
    ui: &dyn PreTurnUi,
) -> usize {
    // L10: Inject identity if conversation is fresh
    autonomy::inject_identity_if_needed(
        messages,
        &ctx.identity.name,
        &ctx.identity.role,
        "nano-agent",
    );

    // L5: Token estimation and memory compaction
    let mut token_count = memory::estimate_tokens(messages);
    if token_count > memory::MICRO_COMPACT_THRESHOLD {
        memory::micro_compact(messages);
    }
    if token_count > memory::THRESHOLD {
        let (new_msgs, path) = memory::auto_compact(messages, llm, transcript_dir);
        *messages = new_msgs;
        ui.compact_notice(&path.display().to_string());
        token_count = memory::estimate_tokens(messages);
    }

    // Render todo state once for nag policy
    let todo_state = ctx
        .services
        .todo
        .read()
        .expect("TodoManager read lock poisoned")
        .render();

    // L2: Nag policy
    {
        let mut nag = ctx.services.nag.lock().expect("NagPolicy lock poisoned");
        nag.tick();
        if nag.should_inject() {
            let nag_msg = format!(
                "[System reminder: You haven't updated your todo list recently. Current state:\n{}]",
                todo_state
            );
            messages.push(serde_json::json!({"role": "user", "content": nag_msg}));
        }
    }

    // L7: Drain background notifications (projected)
    {
        let notifs = ctx
            .services
            .bg
            .lock()
            .expect("BackgroundManager lock poisoned")
            .drain_notifications();
        if !notifs.is_empty() {
            for n in &notifs {
                ui.background_notification(
                    &n.task_id,
                    &format!("{}: {}", n.status, n.result.trim()),
                );
            }
            let text: Vec<String> = notifs
                .iter()
                .map(|n| {
                    format!(
                        "[Background {} {}]: {}",
                        n.task_id,
                        n.status,
                        n.result.trim()
                    )
                })
                .collect();
            let raw = text.join("\n");
            let projected = projector.project("background", &format!("turn_{}", turn_count), &raw);
            messages.push(serde_json::json!({"role": "user", "content": projected}));
        }
    }

    // L8: Check inbox (projected)
    {
        let inbox = ctx.services.message_bus.read_inbox(&ctx.identity.name);
        if !inbox.is_empty() {
            let text = autonomy::format_inbox(&inbox);
            let projected = projector.project("inbox", &format!("turn_{}", turn_count), &text);
            messages.push(serde_json::json!({"role": "user", "content": projected}));
        }
    }

    // Heartbeat: drain cron events (projected)
    {
        let events = ctx.services.heartbeat_manager.drain_events();
        if !events.is_empty() {
            let text = events
                .iter()
                .map(|e| format!("[Cron '{}' fired]: {}", e.name, e.prompt))
                .collect::<Vec<_>>()
                .join("\n");
            let projected =
                projector.project("heartbeat", &format!("turn_{}", turn_count), &text);
            messages.push(serde_json::json!({"role": "user", "content": projected}));
        }
    }

    // Set memory recall query for this turn's seed
    memory_seed.set_query(input);

    token_count
}


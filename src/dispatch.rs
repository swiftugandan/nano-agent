use crate::anthropic::AnthropicLlm;
use crate::autonomy;
use crate::concurrency::LANE_BACKGROUND;
use crate::delegation::SpawnHooks;
use crate::delegation::{child_tools, SubagentFactory};
use crate::delivery;
use crate::handler::{
    exec_err, require_i64, require_str, AgentContext, HandlerError, HandlerRegistry, HandlerResult,
};
use crate::isolation::WorktreeManager;
use crate::openai::OpenAiLlm;
use crate::planning::TodoItem;
use crate::resilience::{AuthProfile, ContextGuard, LlmLogSink, ResilientLlm, RetryPolicy};
use crate::tools;
use crate::types::*;

use std::sync::atomic::Ordering;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// LLM factory
// ---------------------------------------------------------------------------

pub fn create_llm(backend: &str) -> Box<dyn Llm> {
    match backend {
        "openai" => Box::new(OpenAiLlm::from_env()),
        _ => Box::new(AnthropicLlm::from_env()),
    }
}

/// Map an LLM backend name to its auth environment variable prefix.
pub fn auth_prefix_for(backend: &str) -> &'static str {
    match backend {
        "openai" => "OPENROUTER_API_KEY",
        _ => "ANTHROPIC_API_KEY",
    }
}

pub fn wrap_llm(
    inner: Box<dyn Llm>,
    auth_prefix: &str,
    transcript_dir: &std::path::Path,
    log: LlmLogSink,
) -> Box<dyn Llm> {
    let policy = RetryPolicy::from_env();
    let auth = AuthProfile::from_env(auth_prefix);
    let resilient = Box::new(ResilientLlm::new(inner, policy, auth, log.clone()));
    Box::new(ContextGuard::new(resilient, transcript_dir, log))
}

// ---------------------------------------------------------------------------
// Build extended handler registry (L2–L11)
// ---------------------------------------------------------------------------

pub fn build_extended_registry(ctx: &AgentContext) -> HandlerRegistry {
    let mut reg = HandlerRegistry::new();

    // -- L2: todo_update
    reg.register(
        "todo_update",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let items_val = input
                    .get("items")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| HandlerError::Validation {
                        message: "missing 'items' array".into(),
                    })?;
                let items: Vec<TodoItem> = items_val
                    .iter()
                    .filter_map(|v| {
                        Some(TodoItem {
                            id: v.get("id")?.as_str()?.to_string(),
                            text: v.get("text")?.as_str()?.to_string(),
                            status: v.get("status")?.as_str()?.to_string(),
                        })
                    })
                    .collect();
                let mut t = ctx
                    .services
                    .todo
                    .write()
                    .expect("TodoManager write lock poisoned");
                let rendered = t.update(items).map_err(exec_err)?;
                ctx.services
                    .nag
                    .lock()
                    .expect("NagPolicy lock poisoned")
                    .reset();
                Ok(rendered)
            },
        ),
    );

    // -- L2: todo_read
    reg.register(
        "todo_read",
        Arc::new(
            |ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok(ctx
                    .services
                    .todo
                    .read()
                    .expect("TodoManager read lock poisoned")
                    .render())
            },
        ),
    );

    // -- L4: read_skill
    reg.register(
        "read_skill",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let name = require_str(&input, "name")?;
                Ok(ctx.services.skill_loader.get_content(name))
            },
        ),
    );

    // -- L6: task_create
    reg.register(
        "task_create",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let subject = require_str(&input, "subject")?;
                Ok(ctx
                    .services
                    .task_manager
                    .write()
                    .expect("TaskManager write lock poisoned")
                    .create(subject))
            },
        ),
    );

    // -- L6: task_get
    reg.register(
        "task_get",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let task_id = require_i64(&input, "task_id")?;
                Ok(ctx
                    .services
                    .task_manager
                    .read()
                    .expect("TaskManager read lock poisoned")
                    .get(task_id))
            },
        ),
    );

    // -- L6: task_update
    reg.register(
        "task_update",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let task_id = require_i64(&input, "task_id")?;
                let status = input.get("status").and_then(|v| v.as_str());
                let blocked_by: Option<Vec<i64>> = input
                    .get("add_blocked_by")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_i64()).collect());
                let blocks: Option<Vec<i64>> = input
                    .get("add_blocks")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_i64()).collect());
                let tm = ctx
                    .services
                    .task_manager
                    .read()
                    .expect("TaskManager read lock poisoned");
                tm.update(task_id, status, blocked_by.as_deref(), blocks.as_deref())
                    .map_err(exec_err)
            },
        ),
    );

    // -- L6: task_list
    reg.register(
        "task_list",
        Arc::new(
            |ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok(ctx
                    .services
                    .task_manager
                    .read()
                    .expect("TaskManager read lock poisoned")
                    .list_all())
            },
        ),
    );

    // -- L7: background_run
    reg.register(
        "background_run",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let cmd = require_str(&input, "command")?;
                let lane = input
                    .get("lane")
                    .and_then(|v| v.as_str())
                    .unwrap_or(LANE_BACKGROUND);
                Ok(ctx
                    .services
                    .bg
                    .lock()
                    .expect("BackgroundManager lock poisoned")
                    .run_in_lane(lane, cmd))
            },
        ),
    );

    // -- L7: background_check
    reg.register(
        "background_check",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let task_id = require_str(&input, "task_id")?;
                Ok(ctx
                    .services
                    .bg
                    .lock()
                    .expect("BackgroundManager lock poisoned")
                    .check(task_id))
            },
        ),
    );

    // -- L8: send_message
    reg.register(
        "send_message",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let to = require_str(&input, "to")?;
                let content = require_str(&input, "content")?;
                let result = ctx
                    .services
                    .message_bus
                    .send(&ctx.identity.name, to, content);
                let preview: String = content.chars().take(60).collect();
                let preview = if content.len() > 60 {
                    format!("{}...", preview)
                } else {
                    preview
                };
                ctx.services.event_bus.emit_with_data(
                    "delegation_sent",
                    serde_json::json!({ "from": ctx.identity.name, "to": to, "content_preview": preview }),
                );
                Ok(result)
            },
        ),
    );

    // -- L8: broadcast_message
    reg.register(
        "broadcast_message",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let content = require_str(&input, "content")?;
                let names = ctx
                    .services
                    .teammate_manager
                    .read()
                    .expect("TeammateManager read lock poisoned")
                    .member_names();
                let result = ctx
                    .services
                    .message_bus
                    .broadcast(&ctx.identity.name, content, &names);
                let preview: String = content.chars().take(60).collect();
                let preview = if content.len() > 60 {
                    format!("{}...", preview)
                } else {
                    preview
                };
                ctx.services.event_bus.emit_with_data(
                    "delegation_broadcast",
                    serde_json::json!({ "from": ctx.identity.name, "content_preview": preview, "recipients": names.len() }),
                );
                Ok(result)
            },
        ),
    );

    // -- L8: read_inbox
    reg.register(
        "read_inbox",
        Arc::new(
            |ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                let msgs = ctx.services.message_bus.read_inbox(&ctx.identity.name);
                if msgs.is_empty() {
                    Ok("No messages.".into())
                } else {
                    Ok(serde_json::to_string_pretty(&msgs)
                        .unwrap_or_else(|_| "Error reading inbox".into()))
                }
            },
        ),
    );

    // -- L8: list_teammates
    reg.register(
        "list_teammates",
        Arc::new(
            |ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok(ctx
                    .services
                    .teammate_manager
                    .read()
                    .expect("TeammateManager read lock poisoned")
                    .list_all())
            },
        ),
    );

    // -- L8: spawn_teammate (with real lifecycle thread)
    {
        let parent_ctx = ctx.clone();
        reg.register(
            "spawn_teammate",
            Arc::new(move |_ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let name = require_str(&input, "name")?.to_string();
                let role = require_str(&input, "role")?.to_string();
                let prompt = require_str(&input, "prompt")?.to_string();

                // Register teammate
                let result = parent_ctx
                    .services.teammate_manager
                    .write()
                    .expect("TeammateManager write lock poisoned")
                    .spawn(&name, &role, &prompt);
                if result.starts_with("Error") {
                    return Err(HandlerError::Execution { message: result });
                }

                let transcript_d = parent_ctx.cwd.join(".nano-agent").join("transcripts");

                let child_ctx = parent_ctx.child_context(
                    name.clone(),
                    role.clone(),
                    Some(transcript_d.clone()),
                );

                std::thread::spawn(move || {
                    let inner_llm = create_llm(&child_ctx.llm_backend);
                    let mut llm_box = wrap_llm(
                        inner_llm,
                        auth_prefix_for(&child_ctx.llm_backend),
                        &transcript_d,
                        None,
                    );

                    // Build tool defs + registry for the teammate
                    let mut tool_defs = tools::tool_definitions();
                    tool_defs.extend(tools::extended_tool_definitions());

                    let mut registry = tools::build_registry(&child_ctx.cwd);
                    registry.extend(build_extended_registry(&child_ctx));

                    let system = format!(
                        "You are '{}', a teammate agent (role: {}). Your initial task:\n{}\n\n\
                         When you finish your current work, call the `idle` tool to enter idle mode \
                         and wait for new tasks or messages.",
                        child_ctx.identity.name, child_ctx.identity.role, prompt
                    );

                    let mut messages = vec![serde_json::json!({
                        "role": "user",
                        "content": prompt,
                    })];

                    let policy = autonomy::IdlePolicy::default();
                    autonomy::run_teammate_lifecycle(
                        llm_box.as_mut(),
                        &system,
                        &mut messages,
                        &tool_defs,
                        &registry,
                        &child_ctx,
                        &policy,
                    );
                });

                Ok(result)
            }),
        );
    }

    // -- L9: shutdown_teammate
    reg.register(
        "shutdown_teammate",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let t = require_str(&input, "teammate")?;
                Ok(ctx
                    .services
                    .request_tracker
                    .handle_shutdown_request(&ctx.services.message_bus, t))
            },
        ),
    );

    // -- L9: review_plan
    reg.register(
        "review_plan",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let req_id = require_str(&input, "request_id")?;
                let approve = input
                    .get("approve")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let feedback = input.get("feedback").and_then(|v| v.as_str()).unwrap_or("");
                Ok(ctx.services.request_tracker.handle_plan_review(
                    &ctx.services.message_bus,
                    req_id,
                    approve,
                    feedback,
                ))
            },
        ),
    );

    // -- L11: worktree_create
    reg.register(
        "worktree_create",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let name = require_str(&input, "name")?;
                let task_id = input.get("task_id").and_then(|v| v.as_i64());
                let tm_guard = ctx
                    .services
                    .task_manager
                    .write()
                    .expect("TaskManager write lock poisoned");
                let wm = WorktreeManager::new(&ctx.cwd, &tm_guard, &ctx.services.event_bus);
                wm.create_with_task(name, task_id).map_err(exec_err)
            },
        ),
    );

    // -- L11: worktree_remove
    reg.register(
        "worktree_remove",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let name = require_str(&input, "name")?;
                let complete = input
                    .get("complete_task")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let tm_guard = ctx
                    .services
                    .task_manager
                    .write()
                    .expect("TaskManager write lock poisoned");
                let wm = WorktreeManager::new(&ctx.cwd, &tm_guard, &ctx.services.event_bus);
                wm.remove_with_options(name, complete).map_err(exec_err)
            },
        ),
    );

    // -- L11: list_events
    reg.register(
        "list_events",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                Ok(ctx.services.event_bus.list_recent(limit))
            },
        ),
    );

    // -- L10: scan_tasks
    reg.register(
        "scan_tasks",
        Arc::new(
            |ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                let tasks = autonomy::scan_unclaimed_tasks(&ctx.tasks_dir);
                if tasks.is_empty() {
                    Ok("No unclaimed tasks.".into())
                } else {
                    Ok(serde_json::to_string_pretty(&tasks).unwrap_or_else(|_| "Error".into()))
                }
            },
        ),
    );

    // -- L10: claim_task
    reg.register(
        "claim_task",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let id = require_i64(&input, "task_id")?;
                Ok(autonomy::claim_task(&ctx.tasks_dir, id, &ctx.identity.name))
            },
        ),
    );

    // -- L3: subagent
    reg.register(
        "subagent",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let prompt = require_str(&input, "prompt")?;
                let mut child_llm = create_llm(&ctx.llm_backend);
                let child_tool_defs = child_tools();
                let child_registry = tools::build_registry(&ctx.cwd);
                // Subagents only use core tools (no extended services), so a mock
                // context is sufficient. This avoids sharing parent's live services.
                let child_ctx = AgentContext::mock(&ctx.cwd);
                let hooks = SpawnHooks {
                    progress: ctx.subagent_progress.as_ref(),
                    event_bus: Some(&ctx.services.event_bus),
                };
                let out = SubagentFactory::spawn(
                    child_llm.as_mut(),
                    prompt,
                    child_tool_defs,
                    &child_registry,
                    &child_ctx,
                    10,
                    hooks,
                );
                Ok(out)
            },
        ),
    );

    // -- GAP 5: save_memory
    reg.register(
        "save_memory",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let text = require_str(&input, "text")?;
                let tags: Vec<String> = input
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(ctx.services.memory_store.save_memory(text, &tags))
            },
        ),
    );

    // -- GAP 8: enqueue_delivery
    reg.register(
        "enqueue_delivery",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let channel = require_str(&input, "channel")?;
                let peer_id = require_str(&input, "peer_id")?;
                let payload = require_str(&input, "payload")?;
                let max_attempts = input
                    .get("max_attempts")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5) as u32;
                let id = delivery::enqueue_delivery(
                    &ctx.services.delivery_queue,
                    channel,
                    peer_id,
                    payload,
                    max_attempts,
                )
                .map_err(exec_err)?;
                Ok(format!("Delivery enqueued: {}", id))
            },
        ),
    );

    // -- compact tool handler
    reg.register(
        "compact",
        Arc::new(|ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
            ctx.signals.compact.request();
            Ok("Compaction requested. The conversation will be compacted after this tool call completes.".into())
        }),
    );

    // -- idle tool handler
    reg.register(
        "idle",
        Arc::new(
            |ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                ctx.signals.idle.store(true, Ordering::Release);
                Ok("Transitioning to idle phase. You will be woken when new work arrives.".into())
            },
        ),
    );

    // -- cron_add
    reg.register(
        "cron_add",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let name = require_str(&input, "name")?;
                let cron = require_str(&input, "cron")?;
                let prompt = require_str(&input, "prompt")?;
                Ok(ctx.services.heartbeat_manager.add_cron(name, cron, prompt))
            },
        ),
    );

    // -- cron_remove
    reg.register(
        "cron_remove",
        Arc::new(
            |ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let name = require_str(&input, "name")?;
                Ok(ctx.services.heartbeat_manager.remove_cron(name))
            },
        ),
    );

    // -- cron_list
    reg.register(
        "cron_list",
        Arc::new(
            |ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok(ctx.services.heartbeat_manager.list_crons())
            },
        ),
    );

    reg
}

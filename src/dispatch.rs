use crate::anthropic::AnthropicLlm;
use crate::autonomy;
use crate::concurrency::{BackgroundManager, LANE_BACKGROUND};
use crate::delegation::{child_tools, SubagentFactory};
use crate::delivery::{self, DeliveryQueue};
use crate::heartbeat::HeartbeatManager;
use crate::isolation::{EventBus, WorktreeManager};
use crate::knowledge::SkillLoader;
use crate::memory_store::MemoryStore;
use crate::openai::OpenAiLlm;
use crate::planning::{NagPolicy, TodoItem, TodoManager};
use crate::protocols::RequestTracker;
use crate::resilience::{AuthProfile, ContextGuard, ResilientLlm, RetryPolicy};
use crate::tasks::TaskManager;
use crate::teams::{MessageBus, TeammateManager};
use crate::tools;
use crate::types::*;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

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
) -> Box<dyn Llm> {
    let policy = RetryPolicy::from_env();
    let auth = AuthProfile::from_env(auth_prefix);
    let resilient = Box::new(ResilientLlm::new(inner, policy, auth));
    Box::new(ContextGuard::new(resilient, transcript_dir))
}

// ---------------------------------------------------------------------------
// Build extended dispatch closures (L2–L11)
// ---------------------------------------------------------------------------

pub struct AgentConfig {
    pub repo_root: PathBuf,
    pub tasks_dir: PathBuf,
    pub agent_name: String,
    pub llm_backend: String,
}

pub struct Services {
    pub todo: Arc<RwLock<TodoManager>>,
    pub nag: Arc<Mutex<NagPolicy>>,
    pub skill_loader: Arc<SkillLoader>,
    pub task_manager: Arc<RwLock<TaskManager>>,
    pub bg: Arc<Mutex<BackgroundManager>>,
    pub message_bus: Arc<MessageBus>,
    pub teammate_manager: Arc<RwLock<TeammateManager>>,
    pub request_tracker: Arc<RequestTracker>,
    pub event_bus: Arc<EventBus>,
    pub heartbeat_manager: Arc<HeartbeatManager>,
    pub memory_store: Arc<MemoryStore>,
    pub delivery_queue: Arc<DeliveryQueue>,
}

pub struct DispatchContext {
    pub config: AgentConfig,
    pub services: Services,
    pub compact_signal: Arc<CompactSignal>,
    pub idle_signal: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Dispatch macros — reduce boilerplate for common tool handler patterns
// ---------------------------------------------------------------------------

/// Pattern A (str field, write lock): Extract a string field from input, write-lock a service, call a method.
/// Usage: `dispatch_str!(dispatch, "tool_name", "field", service, method, "ServiceType")`
macro_rules! dispatch_str {
    ($dispatch:ident, $tool:expr, $field:expr, $svc:expr, $method:ident, $label:expr) => {{
        let svc = Arc::clone(&$svc);
        $dispatch.insert(
            $tool.into(),
            Box::new(
                move |input| match input.get($field).and_then(|v| v.as_str()) {
                    Some(val) => svc
                        .write()
                        .expect(concat!($label, " write lock poisoned"))
                        .$method(val),
                    None => format!("Error: missing '{}' field", $field),
                },
            ),
        );
    }};
}

/// Pattern A (str field, read lock): Extract a string field from input, read-lock a service, call a method.
/// Usage: `dispatch_str_read!(dispatch, "tool_name", "field", service, method, "ServiceType")`
#[allow(unused_macros)]
macro_rules! dispatch_str_read {
    ($dispatch:ident, $tool:expr, $field:expr, $svc:expr, $method:ident, $label:expr) => {{
        let svc = Arc::clone(&$svc);
        $dispatch.insert(
            $tool.into(),
            Box::new(
                move |input| match input.get($field).and_then(|v| v.as_str()) {
                    Some(val) => svc
                        .read()
                        .expect(concat!($label, " read lock poisoned"))
                        .$method(val),
                    None => format!("Error: missing '{}' field", $field),
                },
            ),
        );
    }};
}

/// Pattern A (i64 field): Extract an i64 field from input, write-lock a service, call a method.
/// Usage: `dispatch_i64!(dispatch, "tool_name", "field", service, method, "ServiceType")`
#[allow(unused_macros)]
macro_rules! dispatch_i64 {
    ($dispatch:ident, $tool:expr, $field:expr, $svc:expr, $method:ident, $label:expr) => {{
        let svc = Arc::clone(&$svc);
        $dispatch.insert(
            $tool.into(),
            Box::new(
                move |input| match input.get($field).and_then(|v| v.as_i64()) {
                    Some(val) => svc
                        .write()
                        .expect(concat!($label, " write lock poisoned"))
                        .$method(val),
                    None => format!("Error: missing '{}' field", $field),
                },
            ),
        );
    }};
}

/// Pattern B (no input): Write-lock a service and call a method with no arguments.
/// Usage: `dispatch_noarg!(dispatch, "tool_name", service, method, "ServiceType")`
#[allow(unused_macros)]
macro_rules! dispatch_noarg {
    ($dispatch:ident, $tool:expr, $svc:expr, $method:ident, $label:expr) => {{
        let svc = Arc::clone(&$svc);
        $dispatch.insert(
            $tool.into(),
            Box::new(move |_| {
                svc.write()
                    .expect(concat!($label, " write lock poisoned"))
                    .$method()
            }),
        );
    }};
}

/// Pattern B-read (no input): Read-lock a service and call a method with no arguments.
/// Usage: `dispatch_noarg_read!(dispatch, "tool_name", service, method, "ServiceType")`
macro_rules! dispatch_noarg_read {
    ($dispatch:ident, $tool:expr, $svc:expr, $method:ident, $label:expr) => {{
        let svc = Arc::clone(&$svc);
        $dispatch.insert(
            $tool.into(),
            Box::new(move |_| {
                svc.read()
                    .expect(concat!($label, " read lock poisoned"))
                    .$method()
            }),
        );
    }};
}

/// Pattern A-read (i64 field): Extract an i64 field from input, read-lock a service, call a method.
/// Usage: `dispatch_i64_read!(dispatch, "tool_name", "field", service, method, "ServiceType")`
macro_rules! dispatch_i64_read {
    ($dispatch:ident, $tool:expr, $field:expr, $svc:expr, $method:ident, $label:expr) => {{
        let svc = Arc::clone(&$svc);
        $dispatch.insert(
            $tool.into(),
            Box::new(
                move |input| match input.get($field).and_then(|v| v.as_i64()) {
                    Some(val) => svc
                        .read()
                        .expect(concat!($label, " read lock poisoned"))
                        .$method(val),
                    None => format!("Error: missing '{}' field", $field),
                },
            ),
        );
    }};
}

/// Pattern B (no input, no lock): Call a method on an Arc<T> with no arguments and no Mutex.
/// Usage: `dispatch_noarg_direct!(dispatch, "tool_name", service, method)`
macro_rules! dispatch_noarg_direct {
    ($dispatch:ident, $tool:expr, $svc:expr, $method:ident) => {{
        let svc = Arc::clone(&$svc);
        $dispatch.insert($tool.into(), Box::new(move |_| svc.$method()));
    }};
}

/// Pattern A (str field, no lock): Extract a string field, call a method on Arc<T> directly.
/// Usage: `dispatch_str_direct!(dispatch, "tool_name", "field", service, method)`
macro_rules! dispatch_str_direct {
    ($dispatch:ident, $tool:expr, $field:expr, $svc:expr, $method:ident) => {{
        let svc = Arc::clone(&$svc);
        $dispatch.insert(
            $tool.into(),
            Box::new(
                move |input| match input.get($field).and_then(|v| v.as_str()) {
                    Some(val) => svc.$method(val),
                    None => format!("Error: missing '{}' field", $field),
                },
            ),
        );
    }};
}

pub fn build_extended_dispatch(ctx: DispatchContext) -> Dispatch {
    let DispatchContext {
        config:
            AgentConfig {
                repo_root,
                tasks_dir,
                agent_name,
                llm_backend,
            },
        services:
            Services {
                todo,
                nag,
                skill_loader,
                task_manager,
                bg,
                message_bus,
                teammate_manager,
                request_tracker,
                event_bus,
                heartbeat_manager,
                memory_store,
                delivery_queue,
            },
        compact_signal,
        idle_signal,
    } = ctx;
    let mut dispatch: Dispatch = HashMap::new();

    // -- L2: todo_update
    {
        let todo = Arc::clone(&todo);
        let nag = Arc::clone(&nag);
        dispatch.insert(
            "todo_update".into(),
            Box::new(move |input| {
                let items_val = match input.get("items").and_then(|v| v.as_array()) {
                    Some(a) => a,
                    None => return "Error: missing 'items' array".into(),
                };
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
                let mut t = todo.write().expect("TodoManager write lock poisoned");
                match t.update(items) {
                    Ok(rendered) => {
                        nag.lock().expect("NagPolicy lock poisoned").reset();
                        rendered
                    }
                    Err(e) => format!("Error: {}", e),
                }
            }),
        );
    }

    // -- L2: todo_read
    dispatch_noarg_read!(dispatch, "todo_read", todo, render, "TodoManager");

    // -- L4: read_skill
    dispatch_str_direct!(dispatch, "read_skill", "name", skill_loader, get_content);

    // -- L6: task_create (writes files + increments AtomicI64, needs write lock)
    dispatch_str!(
        dispatch,
        "task_create",
        "subject",
        task_manager,
        create,
        "TaskManager"
    );

    // -- L6: task_get
    dispatch_i64_read!(
        dispatch,
        "task_get",
        "task_id",
        task_manager,
        get,
        "TaskManager"
    );

    // -- L6: task_update
    {
        let tm = Arc::clone(&task_manager);
        dispatch.insert(
            "task_update".into(),
            Box::new(move |input| {
                let task_id = match input.get("task_id").and_then(|v| v.as_i64()) {
                    Some(id) => id,
                    None => return "Error: missing 'task_id' field".into(),
                };
                let status = input.get("status").and_then(|v| v.as_str());
                let blocked_by: Option<Vec<i64>> = input
                    .get("add_blocked_by")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_i64()).collect());
                let blocks: Option<Vec<i64>> = input
                    .get("add_blocks")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_i64()).collect());
                let tm = tm.read().expect("TaskManager read lock poisoned");
                match tm.update(task_id, status, blocked_by.as_deref(), blocks.as_deref()) {
                    Ok(s) => s,
                    Err(e) => format!("Error: {}", e),
                }
            }),
        );
    }

    // -- L6: task_list
    dispatch_noarg_read!(dispatch, "task_list", task_manager, list_all, "TaskManager");

    // -- L7: background_run
    {
        let bg = Arc::clone(&bg);
        dispatch.insert(
            "background_run".into(),
            Box::new(move |input| {
                let cmd = match input.get("command").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return "Error: missing 'command' field".into(),
                };
                let lane = input
                    .get("lane")
                    .and_then(|v| v.as_str())
                    .unwrap_or(LANE_BACKGROUND);
                bg.lock()
                    .expect("BackgroundManager lock poisoned")
                    .run_in_lane(lane, cmd)
            }),
        );
    }

    // -- L7: background_check
    {
        let svc = Arc::clone(&bg);
        dispatch.insert(
            "background_check".into(),
            Box::new(
                move |input| match input.get("task_id").and_then(|v| v.as_str()) {
                    Some(val) => svc
                        .lock()
                        .expect("BackgroundManager lock poisoned")
                        .check(val),
                    None => "Error: missing 'task_id' field".into(),
                },
            ),
        );
    }

    // -- L8: send_message
    {
        let bus = Arc::clone(&message_bus);
        let name = agent_name.clone();
        dispatch.insert(
            "send_message".into(),
            Box::new(move |input| {
                let to = match input.get("to").and_then(|v| v.as_str()) {
                    Some(t) => t,
                    None => return "Error: missing 'to' field".into(),
                };
                let content = match input.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return "Error: missing 'content' field".into(),
                };
                bus.send(&name, to, content)
            }),
        );
    }

    // -- L8: broadcast_message
    {
        let bus = Arc::clone(&message_bus);
        let tm_mgr = Arc::clone(&teammate_manager);
        let name = agent_name.clone();
        dispatch.insert(
            "broadcast_message".into(),
            Box::new(move |input| {
                let content = match input.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return "Error: missing 'content' field".into(),
                };
                let names = tm_mgr
                    .read()
                    .expect("TeammateManager read lock poisoned")
                    .member_names();
                bus.broadcast(&name, content, &names)
            }),
        );
    }

    // -- L8: read_inbox
    {
        let bus = Arc::clone(&message_bus);
        let name = agent_name.clone();
        dispatch.insert(
            "read_inbox".into(),
            Box::new(move |_| {
                let msgs = bus.read_inbox(&name);
                if msgs.is_empty() {
                    "No messages.".into()
                } else {
                    serde_json::to_string_pretty(&msgs)
                        .unwrap_or_else(|_| "Error reading inbox".into())
                }
            }),
        );
    }

    // -- L8: list_teammates
    dispatch_noarg_read!(
        dispatch,
        "list_teammates",
        teammate_manager,
        list_all,
        "TeammateManager"
    );

    // -- L8: spawn_teammate (with real lifecycle thread)
    {
        let tm_mgr = Arc::clone(&teammate_manager);
        let bus = Arc::clone(&message_bus);
        let tasks_d = tasks_dir.clone();
        let backend = llm_backend.clone();
        let cwd = repo_root.clone();
        let transcript_d = cwd.join(".nano-agent").join("transcripts");
        let sk_loader = Arc::clone(&skill_loader);
        let t_manager = Arc::clone(&task_manager);
        let bg_mgr = Arc::clone(&bg);
        let req_tracker = Arc::clone(&request_tracker);
        let ev_bus = Arc::clone(&event_bus);
        let todo_mgr = Arc::clone(&todo);
        let nag_pol = Arc::clone(&nag);
        let hb_mgr = Arc::clone(&heartbeat_manager);
        let mem_store = Arc::clone(&memory_store);
        let del_queue = Arc::clone(&delivery_queue);
        dispatch.insert(
            "spawn_teammate".into(),
            Box::new(move |input| {
                let name = match input.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n.to_string(),
                    None => return "Error: missing 'name'".into(),
                };
                let role = match input.get("role").and_then(|v| v.as_str()) {
                    Some(r) => r.to_string(),
                    None => return "Error: missing 'role'".into(),
                };
                let prompt = match input.get("prompt").and_then(|v| v.as_str()) {
                    Some(p) => p.to_string(),
                    None => return "Error: missing 'prompt'".into(),
                };

                // Register teammate
                let result = tm_mgr
                    .write()
                    .expect("TeammateManager write lock poisoned")
                    .spawn(&name, &role, &prompt);
                if result.starts_with("Error") {
                    return result;
                }

                // Clone everything needed for the thread
                let tm_mgr_t = Arc::clone(&tm_mgr);
                let bus_t = Arc::clone(&bus);
                let tasks_d_t = tasks_d.clone();
                let transcript_d_t = transcript_d.clone();
                let backend_t = backend.clone();
                let cwd_t = cwd.clone();
                let sk_loader_t = Arc::clone(&sk_loader);
                let t_manager_t = Arc::clone(&t_manager);
                let bg_mgr_t = Arc::clone(&bg_mgr);
                let req_tracker_t = Arc::clone(&req_tracker);
                let ev_bus_t = Arc::clone(&ev_bus);
                let todo_mgr_t = Arc::clone(&todo_mgr);
                let nag_pol_t = Arc::clone(&nag_pol);
                let hb_mgr_t = Arc::clone(&hb_mgr);
                let mem_store_t = Arc::clone(&mem_store);
                let del_queue_t = Arc::clone(&del_queue);
                let name_t = name.clone();
                let role_t = role.clone();
                let prompt_t = prompt.clone();

                std::thread::spawn(move || {
                    let inner_llm = create_llm(&backend_t);
                    let mut llm_box =
                        wrap_llm(inner_llm, auth_prefix_for(&backend_t), &transcript_d_t);

                    // Build tool defs + dispatch for the teammate
                    let mut tool_defs = tools::tool_definitions();
                    tool_defs.extend(tools::extended_tool_definitions());

                    // Teammates only need idle_signal (compact is handled by lifecycle)
                    let teammate_compact = Arc::new(CompactSignal::new());
                    let teammate_idle = Arc::new(AtomicBool::new(false));

                    let mut dispatch = tools::build_dispatch(&cwd_t);
                    let extended = build_extended_dispatch(DispatchContext {
                        config: AgentConfig {
                            repo_root: cwd_t,
                            tasks_dir: tasks_d_t.clone(),
                            agent_name: name_t.clone(),
                            llm_backend: backend_t,
                        },
                        services: Services {
                            todo: todo_mgr_t,
                            nag: nag_pol_t,
                            skill_loader: sk_loader_t,
                            task_manager: t_manager_t,
                            bg: bg_mgr_t,
                            message_bus: Arc::clone(&bus_t),
                            teammate_manager: Arc::clone(&tm_mgr_t),
                            request_tracker: req_tracker_t,
                            event_bus: ev_bus_t,
                            heartbeat_manager: hb_mgr_t,
                            memory_store: mem_store_t,
                            delivery_queue: del_queue_t,
                        },
                        compact_signal: teammate_compact,
                        idle_signal: Arc::clone(&teammate_idle),
                    });
                    dispatch.extend(extended);

                    let system = format!(
                        "You are '{}', a teammate agent (role: {}). Your initial task:\n{}\n\n\
                     When you finish your current work, call the `idle` tool to enter idle mode \
                     and wait for new tasks or messages.",
                        name_t, role_t, prompt_t
                    );

                    let mut messages = vec![serde_json::json!({
                        "role": "user",
                        "content": prompt_t,
                    })];

                    let ctx = autonomy::LifecycleContext {
                        teammate_manager: tm_mgr_t,
                        message_bus: bus_t,
                        tasks_dir: tasks_d_t,
                        transcript_dir: transcript_d_t,
                        agent_name: name_t,
                        idle_signal: teammate_idle,
                    };
                    let config = autonomy::LifecycleConfig::default();
                    autonomy::run_teammate_lifecycle(
                        llm_box.as_mut(),
                        &system,
                        &mut messages,
                        &tool_defs,
                        &dispatch,
                        &ctx,
                        &config,
                    );
                });

                result
            }),
        );
    }

    // -- L9: shutdown_teammate
    {
        let tracker = Arc::clone(&request_tracker);
        let bus = Arc::clone(&message_bus);
        dispatch.insert(
            "shutdown_teammate".into(),
            Box::new(
                move |input| match input.get("teammate").and_then(|v| v.as_str()) {
                    Some(t) => tracker.handle_shutdown_request(&bus, t),
                    None => "Error: missing 'teammate' field".into(),
                },
            ),
        );
    }

    // -- L9: review_plan
    {
        let tracker = Arc::clone(&request_tracker);
        let bus = Arc::clone(&message_bus);
        dispatch.insert(
            "review_plan".into(),
            Box::new(move |input| {
                let req_id = match input.get("request_id").and_then(|v| v.as_str()) {
                    Some(r) => r,
                    None => return "Error: missing 'request_id'".into(),
                };
                let approve = input
                    .get("approve")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let feedback = input.get("feedback").and_then(|v| v.as_str()).unwrap_or("");
                tracker.handle_plan_review(&bus, req_id, approve, feedback)
            }),
        );
    }

    // -- L11: worktree_create
    {
        let tm = Arc::clone(&task_manager);
        let eb = Arc::clone(&event_bus);
        let root = repo_root.clone();
        dispatch.insert(
            "worktree_create".into(),
            Box::new(move |input| {
                let name = match input.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n,
                    None => return "Error: missing 'name'".into(),
                };
                let task_id = input.get("task_id").and_then(|v| v.as_i64());
                let tm_guard = tm.write().expect("TaskManager write lock poisoned");
                let wm = WorktreeManager::new(&root, &tm_guard, &eb);
                match wm.create_with_task(name, task_id) {
                    Ok(s) => s,
                    Err(e) => format!("Error: {}", e),
                }
            }),
        );
    }

    // -- L11: worktree_remove
    {
        let tm = Arc::clone(&task_manager);
        let eb = Arc::clone(&event_bus);
        let root = repo_root.clone();
        dispatch.insert(
            "worktree_remove".into(),
            Box::new(move |input| {
                let name = match input.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n,
                    None => return "Error: missing 'name'".into(),
                };
                let complete = input
                    .get("complete_task")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let tm_guard = tm.write().expect("TaskManager write lock poisoned");
                let wm = WorktreeManager::new(&root, &tm_guard, &eb);
                match wm.remove_with_options(name, complete) {
                    Ok(s) => s,
                    Err(e) => format!("Error: {}", e),
                }
            }),
        );
    }

    // -- L11: list_events
    {
        let eb = Arc::clone(&event_bus);
        dispatch.insert(
            "list_events".into(),
            Box::new(move |input| {
                let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                eb.list_recent(limit)
            }),
        );
    }

    // -- L10: scan_tasks
    {
        let dir = tasks_dir.clone();
        dispatch.insert(
            "scan_tasks".into(),
            Box::new(move |_| {
                let tasks = autonomy::scan_unclaimed_tasks(&dir);
                if tasks.is_empty() {
                    "No unclaimed tasks.".into()
                } else {
                    serde_json::to_string_pretty(&tasks).unwrap_or_else(|_| "Error".into())
                }
            }),
        );
    }

    // -- L10: claim_task
    {
        let dir = tasks_dir.clone();
        let name = agent_name.clone();
        dispatch.insert(
            "claim_task".into(),
            Box::new(
                move |input| match input.get("task_id").and_then(|v| v.as_i64()) {
                    Some(id) => autonomy::claim_task(&dir, id, &name),
                    None => "Error: missing 'task_id'".into(),
                },
            ),
        );
    }

    // -- L3: subagent
    {
        let backend = llm_backend;
        let cwd = repo_root;
        dispatch.insert(
            "subagent".into(),
            Box::new(move |input| {
                let prompt = match input.get("prompt").and_then(|v| v.as_str()) {
                    Some(p) => p,
                    None => return "Error: missing 'prompt'".into(),
                };
                let mut child_llm = create_llm(&backend);
                let child_tool_defs = child_tools();
                let child_dispatch = tools::build_dispatch(&cwd);
                SubagentFactory::spawn(
                    child_llm.as_mut(),
                    prompt,
                    child_tool_defs,
                    &child_dispatch,
                    10,
                )
            }),
        );
    }

    // -- GAP 5: save_memory
    {
        let ms = Arc::clone(&memory_store);
        dispatch.insert(
            "save_memory".into(),
            Box::new(move |input| {
                let text = match input.get("text").and_then(|v| v.as_str()) {
                    Some(t) => t,
                    None => return "Error: missing 'text' field".into(),
                };
                let tags: Vec<String> = input
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                ms.save_memory(text, &tags)
            }),
        );
    }

    // -- GAP 8: enqueue_delivery
    {
        let dq = Arc::clone(&delivery_queue);
        dispatch.insert(
            "enqueue_delivery".into(),
            Box::new(move |input| {
                let channel = match input.get("channel").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return "Error: missing 'channel' field".into(),
                };
                let peer_id = match input.get("peer_id").and_then(|v| v.as_str()) {
                    Some(p) => p,
                    None => return "Error: missing 'peer_id' field".into(),
                };
                let payload = match input.get("payload").and_then(|v| v.as_str()) {
                    Some(p) => p,
                    None => return "Error: missing 'payload' field".into(),
                };
                let max_attempts = input
                    .get("max_attempts")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5) as u32;
                match delivery::enqueue_delivery(&dq, channel, peer_id, payload, max_attempts) {
                    Ok(id) => format!("Delivery enqueued: {}", id),
                    Err(e) => format!("Error: {}", e),
                }
            }),
        );
    }

    // -- compact tool handler
    {
        let signal = compact_signal;
        dispatch.insert("compact".into(), Box::new(move |_| {
            signal.request();
            "Compaction requested. The conversation will be compacted after this tool call completes.".into()
        }));
    }

    // -- idle tool handler
    {
        let signal = idle_signal;
        dispatch.insert(
            "idle".into(),
            Box::new(move |_| {
                signal.store(true, Ordering::Release);
                "Transitioning to idle phase. You will be woken when new work arrives.".into()
            }),
        );
    }

    // -- cron_add
    {
        let hb = Arc::clone(&heartbeat_manager);
        dispatch.insert(
            "cron_add".into(),
            Box::new(move |input| {
                let name = match input.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n,
                    None => return "Error: missing 'name'".into(),
                };
                let cron = match input.get("cron").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return "Error: missing 'cron'".into(),
                };
                let prompt = match input.get("prompt").and_then(|v| v.as_str()) {
                    Some(p) => p,
                    None => return "Error: missing 'prompt'".into(),
                };
                hb.add_cron(name, cron, prompt)
            }),
        );
    }

    // -- cron_remove
    dispatch_str_direct!(
        dispatch,
        "cron_remove",
        "name",
        heartbeat_manager,
        remove_cron
    );

    // -- cron_list
    dispatch_noarg_direct!(dispatch, "cron_list", heartbeat_manager, list_crons);

    dispatch
}

use nano_agent::anthropic::AnthropicLlm;
use nano_agent::autonomy;
use nano_agent::concurrency::BackgroundManager;
use nano_agent::core_loop::run_agent_loop;
use nano_agent::delegation::{child_tools, SubagentFactory};
use nano_agent::heartbeat::HeartbeatManager;
use nano_agent::isolation::{EventBus, WorktreeManager};
use nano_agent::knowledge::SkillLoader;
use nano_agent::memory;
use nano_agent::openai::OpenAiLlm;
use nano_agent::planning::{NagPolicy, TodoItem, TodoManager};
use nano_agent::prompt::{PromptAssembler, PromptContext};
use nano_agent::protocols::RequestTracker;
use nano_agent::resilience::{AuthProfile, ResilientLlm, RetryPolicy};
use nano_agent::tasks::TaskManager;
use nano_agent::teams::{MessageBus, TeammateManager};
use nano_agent::tools;
use nano_agent::types::*;

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// LLM factory
// ---------------------------------------------------------------------------

fn create_llm(backend: &str) -> Box<dyn Llm> {
    match backend {
        "openai" => Box::new(OpenAiLlm::from_env()),
        _ => Box::new(AnthropicLlm::from_env()),
    }
}

// ---------------------------------------------------------------------------
// Extended tool definitions (L2–L11)
// ---------------------------------------------------------------------------

fn extended_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "name": "todo_update",
            "description": "Update the todo list. Only one item may be in_progress at a time.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":     { "type": "string" },
                                "text":   { "type": "string" },
                                "status": { "type": "string", "enum": ["pending","in_progress","completed"] }
                            },
                            "required": ["id","text","status"]
                        }
                    }
                },
                "required": ["items"]
            }
        }),
        serde_json::json!({
            "name": "todo_read",
            "description": "Read the current todo list.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "read_skill",
            "description": "Load a skill by name and return its full content.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Skill name" }
                },
                "required": ["name"]
            }
        }),
        serde_json::json!({
            "name": "task_create",
            "description": "Create a new task with a subject.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string" }
                },
                "required": ["subject"]
            }
        }),
        serde_json::json!({
            "name": "task_get",
            "description": "Get a task by ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "integer" }
                },
                "required": ["task_id"]
            }
        }),
        serde_json::json!({
            "name": "task_update",
            "description": "Update a task's status or dependencies.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id":        { "type": "integer" },
                    "status":         { "type": "string", "enum": ["pending","in_progress","completed"] },
                    "add_blocked_by": { "type": "array", "items": { "type": "integer" } },
                    "add_blocks":     { "type": "array", "items": { "type": "integer" } }
                },
                "required": ["task_id"]
            }
        }),
        serde_json::json!({
            "name": "task_list",
            "description": "List all tasks with their status and dependencies.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "background_run",
            "description": "Run a shell command in the background. Returns a task ID immediately.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "lane": { "type": "string", "description": "Lane name (default: 'background'). Available: main, cron, background" }
                },
                "required": ["command"]
            }
        }),
        serde_json::json!({
            "name": "background_check",
            "description": "Check the status and result of a background task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" }
                },
                "required": ["task_id"]
            }
        }),
        serde_json::json!({
            "name": "send_message",
            "description": "Send a message to a teammate via the message bus.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to":      { "type": "string", "description": "Recipient name" },
                    "content": { "type": "string" }
                },
                "required": ["to","content"]
            }
        }),
        serde_json::json!({
            "name": "broadcast_message",
            "description": "Broadcast a message to all teammates.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"]
            }
        }),
        serde_json::json!({
            "name": "read_inbox",
            "description": "Read and clear your inbox messages.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "list_teammates",
            "description": "List all team members and their roles.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "spawn_teammate",
            "description": "Create or reactivate a teammate.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":   { "type": "string" },
                    "role":   { "type": "string" },
                    "prompt": { "type": "string" }
                },
                "required": ["name","role","prompt"]
            }
        }),
        serde_json::json!({
            "name": "shutdown_teammate",
            "description": "Request a teammate to shut down (requires their approval).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "teammate": { "type": "string" }
                },
                "required": ["teammate"]
            }
        }),
        serde_json::json!({
            "name": "review_plan",
            "description": "Approve or reject a pending plan request.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "request_id": { "type": "string" },
                    "approve":    { "type": "boolean" },
                    "feedback":   { "type": "string" }
                },
                "required": ["request_id","approve","feedback"]
            }
        }),
        serde_json::json!({
            "name": "worktree_create",
            "description": "Create a git worktree, optionally bound to a task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":    { "type": "string", "description": "Worktree name (letters, numbers, .-_)" },
                    "task_id": { "type": "integer", "description": "Optional task to bind" }
                },
                "required": ["name"]
            }
        }),
        serde_json::json!({
            "name": "worktree_remove",
            "description": "Remove a git worktree.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":          { "type": "string" },
                    "complete_task": { "type": "boolean", "description": "Mark bound task as completed" }
                },
                "required": ["name"]
            }
        }),
        serde_json::json!({
            "name": "list_events",
            "description": "List recent events from the event log.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "description": "Number of events (default 20)" }
                }
            }
        }),
        serde_json::json!({
            "name": "scan_tasks",
            "description": "Scan for unclaimed tasks (pending, no owner, not blocked).",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "claim_task",
            "description": "Claim an unclaimed task, setting yourself as owner.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "integer" }
                },
                "required": ["task_id"]
            }
        }),
        serde_json::json!({
            "name": "subagent",
            "description": "Spawn an isolated subagent to handle a subtask. Returns only its final text output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "Task description for the subagent" }
                },
                "required": ["prompt"]
            }
        }),
        serde_json::json!({
            "name": "compact",
            "description": "Trigger on-demand conversation compaction. Saves the current transcript and replaces messages with a summary. Use when the conversation is getting long and you want to free up context.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "idle",
            "description": "Transition from WORK to IDLE phase. Use when you have completed your current task and are waiting for new work. Only available to teammates, not the lead agent.",
            "input_schema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "cron_add",
            "description": "Add a scheduled cron job. The prompt will be injected as a user message when the cron fires.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":   { "type": "string", "description": "Unique name for this cron entry" },
                    "cron":   { "type": "string", "description": "Cron expression: minute hour day month weekday (e.g. '*/5 * * * *')" },
                    "prompt": { "type": "string", "description": "Prompt to inject when cron fires" }
                },
                "required": ["name", "cron", "prompt"]
            }
        }),
        serde_json::json!({
            "name": "cron_remove",
            "description": "Remove a scheduled cron job by name.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                },
                "required": ["name"]
            }
        }),
        serde_json::json!({
            "name": "cron_list",
            "description": "List all cron entries with their schedules and enabled status.",
            "input_schema": { "type": "object", "properties": {} }
        }),
    ]
}

// ---------------------------------------------------------------------------
// Build extended dispatch closures (L2–L11)
// ---------------------------------------------------------------------------

fn build_extended_dispatch(
    todo: Arc<Mutex<TodoManager>>,
    nag: Arc<Mutex<NagPolicy>>,
    skill_loader: Arc<SkillLoader>,
    task_manager: Arc<Mutex<TaskManager>>,
    bg: Arc<Mutex<BackgroundManager>>,
    message_bus: Arc<MessageBus>,
    teammate_manager: Arc<Mutex<TeammateManager>>,
    request_tracker: Arc<RequestTracker>,
    event_bus: Arc<EventBus>,
    heartbeat_manager: Arc<HeartbeatManager>,
    repo_root: PathBuf,
    tasks_dir: PathBuf,
    agent_name: String,
    llm_backend: String,
    compact_signal: Arc<CompactSignal>,
    idle_signal: Arc<AtomicBool>,
) -> Dispatch {
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
                let mut t = todo.lock().unwrap();
                match t.update(items) {
                    Ok(rendered) => {
                        nag.lock().unwrap().reset();
                        rendered
                    }
                    Err(e) => format!("Error: {}", e),
                }
            }),
        );
    }

    // -- L2: todo_read
    {
        let todo = Arc::clone(&todo);
        dispatch.insert(
            "todo_read".into(),
            Box::new(move |_| todo.lock().unwrap().render()),
        );
    }

    // -- L4: read_skill
    {
        let loader = Arc::clone(&skill_loader);
        dispatch.insert(
            "read_skill".into(),
            Box::new(
                move |input| match input.get("name").and_then(|v| v.as_str()) {
                    Some(name) => loader.get_content(name),
                    None => "Error: missing 'name' field".into(),
                },
            ),
        );
    }

    // -- L6: task_create
    {
        let tm = Arc::clone(&task_manager);
        dispatch.insert(
            "task_create".into(),
            Box::new(
                move |input| match input.get("subject").and_then(|v| v.as_str()) {
                    Some(subject) => tm.lock().unwrap().create(subject),
                    None => "Error: missing 'subject' field".into(),
                },
            ),
        );
    }

    // -- L6: task_get
    {
        let tm = Arc::clone(&task_manager);
        dispatch.insert(
            "task_get".into(),
            Box::new(
                move |input| match input.get("task_id").and_then(|v| v.as_i64()) {
                    Some(id) => tm.lock().unwrap().get(id),
                    None => "Error: missing 'task_id' field".into(),
                },
            ),
        );
    }

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
                let tm = tm.lock().unwrap();
                match tm.update(task_id, status, blocked_by.as_deref(), blocks.as_deref()) {
                    Ok(s) => s,
                    Err(e) => format!("Error: {}", e),
                }
            }),
        );
    }

    // -- L6: task_list
    {
        let tm = Arc::clone(&task_manager);
        dispatch.insert(
            "task_list".into(),
            Box::new(move |_| tm.lock().unwrap().list_all()),
        );
    }

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
                    .unwrap_or("background");
                bg.lock().unwrap().run_in_lane(lane, cmd)
            }),
        );
    }

    // -- L7: background_check
    {
        let bg = Arc::clone(&bg);
        dispatch.insert(
            "background_check".into(),
            Box::new(
                move |input| match input.get("task_id").and_then(|v| v.as_str()) {
                    Some(id) => bg.lock().unwrap().check(id),
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
                let names = tm_mgr.lock().unwrap().member_names();
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
    {
        let tm_mgr = Arc::clone(&teammate_manager);
        dispatch.insert(
            "list_teammates".into(),
            Box::new(move |_| tm_mgr.lock().unwrap().list_all()),
        );
    }

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
                let result = tm_mgr.lock().unwrap().spawn(&name, &role, &prompt);
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
                let name_t = name.clone();
                let role_t = role.clone();
                let prompt_t = prompt.clone();

                std::thread::spawn(move || {
                    let inner_llm = create_llm(&backend_t);
                    let policy = RetryPolicy::from_env();
                    let auth_prefix = if backend_t == "openai" {
                        "OPENROUTER_API_KEY"
                    } else {
                        "ANTHROPIC_API_KEY"
                    };
                    let auth = AuthProfile::from_env(auth_prefix);
                    let mut llm_box: Box<dyn Llm> =
                        Box::new(ResilientLlm::new(inner_llm, policy, auth));

                    // Build tool defs + dispatch for the teammate
                    let mut tool_defs = tools::tool_definitions();
                    tool_defs.extend(extended_tool_definitions());

                    // Teammates only need idle_signal (compact is handled by lifecycle)
                    let teammate_compact = Arc::new(CompactSignal::new());
                    let teammate_idle = Arc::new(AtomicBool::new(false));

                    let mut dispatch = tools::build_dispatch(&cwd_t);
                    let extended = build_extended_dispatch(
                        todo_mgr_t,
                        nag_pol_t,
                        sk_loader_t,
                        t_manager_t,
                        bg_mgr_t,
                        Arc::clone(&bus_t),
                        Arc::clone(&tm_mgr_t),
                        req_tracker_t,
                        ev_bus_t,
                        hb_mgr_t,
                        cwd_t,
                        tasks_d_t.clone(),
                        name_t.clone(),
                        backend_t,
                        teammate_compact,
                        Arc::clone(&teammate_idle),
                    );
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
                    };
                    let config = autonomy::LifecycleConfig::default();
                    autonomy::run_teammate_lifecycle(
                        llm_box.as_mut(),
                        &system,
                        &mut messages,
                        &tool_defs,
                        &dispatch,
                        &ctx,
                        &teammate_idle,
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
                let tm_guard = tm.lock().unwrap();
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
                let tm_guard = tm.lock().unwrap();
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
    {
        let hb = Arc::clone(&heartbeat_manager);
        dispatch.insert(
            "cron_remove".into(),
            Box::new(
                move |input| match input.get("name").and_then(|v| v.as_str()) {
                    Some(name) => hb.remove_cron(name),
                    None => "Error: missing 'name'".into(),
                },
            ),
        );
    }

    // -- cron_list
    {
        let hb = Arc::clone(&heartbeat_manager);
        dispatch.insert("cron_list".into(), Box::new(move |_| hb.list_crons()));
    }

    dispatch
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let backend = std::env::var("LLM_BACKEND").unwrap_or_else(|_| "anthropic".into());
    let cwd = std::env::current_dir().expect("Failed to get current directory");
    let data_dir = cwd.join(".nano-agent");
    std::fs::create_dir_all(&data_dir).ok();

    // -- LLM backend (wrapped in resilience layer)
    let (mut llm_box, model_name): (Box<dyn Llm>, String) = match backend.as_str() {
        "openai" => {
            let llm = OpenAiLlm::from_env();
            let model = llm.model.clone();
            let policy = RetryPolicy::from_env();
            let auth = AuthProfile::from_env("OPENROUTER_API_KEY");
            (
                Box::new(ResilientLlm::new(Box::new(llm), policy, auth)),
                model,
            )
        }
        _ => {
            let llm = AnthropicLlm::from_env();
            let model = llm.model.clone();
            let policy = RetryPolicy::from_env();
            let auth = AuthProfile::from_env("ANTHROPIC_API_KEY");
            (
                Box::new(ResilientLlm::new(Box::new(llm), policy, auth)),
                model,
            )
        }
    };

    // -- Data directories
    let tasks_dir = data_dir.join("tasks");
    let transcript_dir = data_dir.join("transcripts");
    let inbox_dir = data_dir.join("inbox");
    let team_dir = data_dir.join("team");
    let events_path = data_dir.join("events.jsonl");
    let skills_dir = std::env::var("SKILLS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("skills"));

    // -- Agent identity
    let agent_name = std::env::var("AGENT_NAME").unwrap_or_else(|_| "lead".into());
    let agent_role = std::env::var("AGENT_ROLE").unwrap_or_else(|_| "developer".into());

    // -- Initialize all managers
    let task_manager = Arc::new(Mutex::new(TaskManager::new(&tasks_dir)));
    let event_bus = Arc::new(EventBus::new(&events_path));
    let skill_loader = Arc::new(SkillLoader::new(&skills_dir));
    let transcript_store = memory::TranscriptStore::new(&transcript_dir);
    let background_manager = Arc::new(Mutex::new(BackgroundManager::new()));
    let todo_manager = Arc::new(Mutex::new(TodoManager::new()));
    let nag_policy = Arc::new(Mutex::new(NagPolicy::new(3)));
    let message_bus = Arc::new(MessageBus::new(&inbox_dir));
    let teammate_manager = Arc::new(Mutex::new(TeammateManager::new(&team_dir)));
    let request_tracker = Arc::new(RequestTracker::new());

    // -- Heartbeat / cron scheduler
    let cron_path = data_dir.join("cron.json");
    let heartbeat_manager = Arc::new(HeartbeatManager::new(&cron_path));
    heartbeat_manager.start();

    // -- Prompt assembler
    let mut prompt_assembler = PromptAssembler::new(&data_dir.join("prompts"));
    prompt_assembler.init_defaults();

    // -- Register self as a teammate
    teammate_manager
        .lock()
        .unwrap()
        .spawn(&agent_name, &agent_role, "primary agent");
    // Pre-warm: list team (ensures config is written)
    let _ = teammate_manager.lock().unwrap().list_all();

    // -- Signals for compact and idle
    let compact_signal = Arc::new(CompactSignal::new());
    let idle_signal = Arc::new(AtomicBool::new(false));

    // -- Build tool definitions: base (4) + extended (24)
    let mut tool_defs = tools::tool_definitions();
    tool_defs.extend(extended_tool_definitions());

    // -- Build dispatch: base + extended
    let mut dispatch = tools::build_dispatch(&cwd);
    let extended = build_extended_dispatch(
        Arc::clone(&todo_manager),
        Arc::clone(&nag_policy),
        Arc::clone(&skill_loader),
        Arc::clone(&task_manager),
        Arc::clone(&background_manager),
        Arc::clone(&message_bus),
        Arc::clone(&teammate_manager),
        Arc::clone(&request_tracker),
        Arc::clone(&event_bus),
        Arc::clone(&heartbeat_manager),
        cwd.to_path_buf(),
        tasks_dir.clone(),
        agent_name.clone(),
        backend.clone(),
        Arc::clone(&compact_signal),
        Arc::clone(&idle_signal),
    );
    dispatch.extend(extended);

    // -- Startup banner
    println!("╔══════════════════════════════════════════════╗");
    println!("║  nano-agent v0.1.0                          ║");
    println!(
        "║  backend: {:35}║",
        format!("{} ({})", backend, model_name)
    );
    println!(
        "║  agent:   {:35}║",
        format!("{} [{}]", agent_name, agent_role)
    );
    println!(
        "║  tools:   {:35}║",
        format!("{} registered", tool_defs.len())
    );
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("Commands: /quit  /clear  /status  /tasks  /team  /events");
    println!();

    let mut messages: Vec<serde_json::Value> = Vec::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // -- Slash commands
        match input {
            "/quit" => break,
            "/clear" => {
                transcript_store.save(&messages);
                messages.clear();
                println!("[cleared] Transcript saved.\n");
                continue;
            }
            "/status" => {
                let tokens = memory::estimate_tokens(&messages);
                let todo_state = todo_manager.lock().unwrap().render();
                let bg_notifs = background_manager.lock().unwrap().drain_notifications();
                println!("  tokens:  ~{}", tokens);
                println!("  turns:   {}", messages.len());
                println!("  todo:\n{}", todo_state);
                if !bg_notifs.is_empty() {
                    println!("  background: {} completed", bg_notifs.len());
                }
                println!();
                continue;
            }
            "/tasks" => {
                println!("{}\n", task_manager.lock().unwrap().list_all());
                continue;
            }
            "/team" => {
                println!("{}\n", teammate_manager.lock().unwrap().list_all());
                continue;
            }
            "/events" => {
                println!("{}\n", event_bus.list_recent(20));
                continue;
            }
            _ => {}
        }

        // -- Add user message
        messages.push(serde_json::json!({
            "role": "user",
            "content": input,
        }));

        // -- Pre-turn pipeline

        // L10: Inject identity if conversation is fresh
        if autonomy::should_inject(&messages) {
            let identity = autonomy::make_identity_block(&agent_name, &agent_role, "nano-agent");
            messages.insert(0, identity);
        }

        // L5: Token estimation and memory compaction
        let token_count = memory::estimate_tokens(&messages);
        if token_count > 80_000 {
            memory::micro_compact(&mut messages);
        }
        if token_count > memory::THRESHOLD {
            let (new_msgs, path) =
                memory::auto_compact(&messages, llm_box.as_mut(), &transcript_dir);
            messages = new_msgs;
            println!("[compact] Saved transcript: {}", path.display());
        }

        // Render todo state once for both nag policy and prompt assembly
        let todo_state = todo_manager.lock().unwrap().render();

        // L2: Nag policy
        {
            let mut nag = nag_policy.lock().unwrap();
            nag.tick();
            if nag.should_inject() {
                let nag_msg = format!(
                    "[System reminder: You haven't updated your todo list recently. Current state:\n{}]",
                    todo_state
                );
                messages.push(serde_json::json!({"role": "user", "content": nag_msg}));
            }
        }

        // L7: Drain background notifications
        {
            let notifs = background_manager.lock().unwrap().drain_notifications();
            if !notifs.is_empty() {
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
                messages.push(serde_json::json!({"role": "user", "content": text.join("\n")}));
            }
        }

        // L8: Check inbox
        {
            let inbox = message_bus.read_inbox(&agent_name);
            if !inbox.is_empty() {
                let text = autonomy::format_inbox(&inbox);
                messages.push(serde_json::json!({"role": "user", "content": text}));
            }
        }

        // -- Heartbeat: drain cron events
        {
            let events = heartbeat_manager.drain_events();
            if !events.is_empty() {
                let text = events
                    .iter()
                    .map(|e| format!("[Cron '{}' fired]: {}", e.name, e.prompt))
                    .collect::<Vec<_>>()
                    .join("\n");
                messages.push(serde_json::json!({"role": "user", "content": text}));
            }
        }

        // -- Build dynamic system prompt via assembler
        let skill_desc = skill_loader.get_descriptions();
        prompt_assembler.reload();
        let system = prompt_assembler.compose(&PromptContext {
            agent_name: agent_name.clone(),
            agent_role: agent_role.clone(),
            cwd: cwd.display().to_string(),
            tool_count: tool_defs.len(),
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
        });

        // -- Run the agent loop
        let signals = LoopSignals {
            compact_signal: Some(&compact_signal),
            transcript_dir: Some(&transcript_dir),
            idle_signal: None, // no idle for lead agent
        };
        let _calls = run_agent_loop(
            llm_box.as_mut(),
            &system,
            &mut messages,
            &tool_defs,
            &dispatch,
            &signals,
        );

        // -- Print the last assistant message
        if let Some(last) = messages.last() {
            if let Some(content) = last.get("content") {
                if let Some(arr) = content.as_array() {
                    for block in arr {
                        if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                println!("\n{}\n", text);
                            }
                        }
                    }
                } else if let Some(text) = content.as_str() {
                    println!("\n{}\n", text);
                }
            }
        }
    }

    // -- Save transcript on exit
    if !messages.is_empty() {
        transcript_store.save(&messages);
        println!("[exit] Transcript saved.");
    }
}

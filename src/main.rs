use clap::Parser;
use nano_agent::anthropic::AnthropicLlm;
use nano_agent::autonomy;
use nano_agent::channels::{ChannelManager, CliChannel};
use nano_agent::cli::Cli;
use nano_agent::concurrency::BackgroundManager;
use nano_agent::context::{MemorySeed, Projector, SeedCollector, SkillSeed, TaskSeed, TodoSeed};
use nano_agent::core_loop::run_agent_loop;
use nano_agent::delivery::{DeliveryQueue, DeliveryRunner};
use nano_agent::dispatch::{
    auth_prefix_for, build_extended_dispatch, wrap_llm, AgentConfig, DispatchContext, Services,
};
use nano_agent::gateway::Gateway;
use nano_agent::heartbeat::HeartbeatManager;
use nano_agent::isolation::EventBus;
use nano_agent::knowledge::SkillLoader;
use nano_agent::memory;
use nano_agent::memory_store::MemoryStore;
use nano_agent::openai::OpenAiLlm;
use nano_agent::planning::{NagPolicy, TodoManager};
use nano_agent::prompt::{build_prompt_context, extract_last_response_text, PromptAssembler};
use nano_agent::protocols::RequestTracker;
use nano_agent::tasks::TaskManager;
use nano_agent::teams::{MessageBus, TeammateManager};
use nano_agent::tools;
use nano_agent::types::*;
use nano_agent::ui::{NanoPrompt, PromptState, SpinnerHandle, UiRenderer};

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let backend = std::env::var("LLM_BACKEND").unwrap_or_else(|_| "anthropic".into());
    let cwd = std::env::current_dir().expect("Failed to get current directory");
    let data_dir = cwd.join(".nano-agent");
    std::fs::create_dir_all(&data_dir).ok();

    // -- Detect piped stdin and build oneshot prompt
    let piped_input = if !std::io::stdin().is_terminal() {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf).ok();
        if buf.trim().is_empty() {
            None
        } else {
            Some(buf)
        }
    } else {
        None
    };

    let oneshot_prompt = match (&cli.prompt, &piped_input) {
        (Some(arg), Some(stdin)) => Some(format!("{}\n\n{}", arg, stdin)),
        (Some(arg), None) => Some(arg.clone()),
        (None, Some(stdin)) => Some(stdin.clone()),
        (None, None) => None,
    };
    let oneshot = oneshot_prompt.is_some();

    // -- LLM backend (wrapped in resilience + ContextGuard layers)
    let transcript_dir = data_dir.join("transcripts");
    let (mut llm_box, model_name): (Box<dyn Llm>, String) = {
        let (inner, model) = match backend.as_str() {
            "openai" => {
                let llm = OpenAiLlm::from_env();
                let m = llm.model.clone();
                (Box::new(llm) as Box<dyn Llm>, m)
            }
            _ => {
                let llm = AnthropicLlm::from_env();
                let m = llm.model.clone();
                (Box::new(llm) as Box<dyn Llm>, m)
            }
        };
        (
            wrap_llm(inner, auth_prefix_for(&backend), &transcript_dir),
            model,
        )
    };

    // -- Data directories
    let tasks_dir = data_dir.join("tasks");
    let session_dir = data_dir.join("sessions");
    let inbox_dir = data_dir.join("inbox");
    let team_dir = data_dir.join("team");
    let events_path = data_dir.join("events.jsonl");
    let delivery_dir = data_dir.join("delivery");
    let memories_dir = data_dir.join("memories");
    let skills_dir = std::env::var("SKILLS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("skills"));

    // -- Agent identity
    let agent_name = std::env::var("AGENT_NAME").unwrap_or_else(|_| "lead".into());
    let agent_role = std::env::var("AGENT_ROLE").unwrap_or_else(|_| "developer".into());
    let agent_id = uuid::Uuid::new_v4().to_string();

    // -- GAP 2: Session management
    let session_id = std::env::var("RESUME_SESSION")
        .ok()
        .or(cli.resume)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let session_store = memory::SessionStore::new(&session_dir, &session_id);

    // -- Evict old data files at startup (skip in oneshot to avoid unnecessary I/O)
    if !oneshot {
        memory::SessionStore::evict_old_sessions(&session_dir, 30);
    }

    // -- Initialize all managers
    let task_manager = Arc::new(RwLock::new(TaskManager::new(&tasks_dir)));
    let event_bus = Arc::new(EventBus::new(&events_path));
    let skill_loader = Arc::new(SkillLoader::new(&skills_dir));
    let transcript_store = memory::TranscriptStore::new(&transcript_dir);
    let background_manager = Arc::new(Mutex::new(BackgroundManager::new()));
    let todo_manager = Arc::new(RwLock::new(TodoManager::new()));
    let nag_policy = Arc::new(Mutex::new(NagPolicy::new(3)));
    let message_bus = Arc::new(MessageBus::new(&inbox_dir));
    let teammate_manager = Arc::new(RwLock::new(TeammateManager::new(&team_dir)));
    let request_tracker = Arc::new(RequestTracker::new());

    // -- GAP 5: Memory store with TF-IDF search
    let memory_store_inner =
        MemoryStore::new(&data_dir.join("prompts").join("MEMORY.md"), &memories_dir);
    if !oneshot {
        memory_store_inner.evict_old(90);
    }
    let memory_store = Arc::new(memory_store_inner);

    // -- GAP 8: Delivery queue
    let delivery_queue = Arc::new(DeliveryQueue::new(&delivery_dir));

    // -- Heartbeat / cron scheduler
    let cron_path = data_dir.join("cron.json");
    let heartbeat_manager = Arc::new(HeartbeatManager::new(&cron_path));

    // -- GAP 6: Channel manager (REPL-only)
    let channel_manager = Arc::new(Mutex::new(ChannelManager::new()));

    // -- Prompt state (shared between NanoPrompt and main loop for updating turn/token counts)
    let prompt_state = Arc::new(Mutex::new(PromptState::new(&agent_name, &agent_role)));

    // -- CLI channel (reedline-based, owned separately for direct read_line access)
    let cli_channel: Option<Arc<CliChannel>> = if !oneshot {
        let history_path = data_dir.join("history.txt");
        let nano_prompt = NanoPrompt::new(Arc::clone(&prompt_state));
        let cli_ch = Arc::new(
            CliChannel::new(nano_prompt, &history_path).expect("Failed to create CLI channel"),
        );
        channel_manager
            .lock()
            .expect("ChannelManager lock poisoned")
            .add(Arc::clone(&cli_ch) as Arc<dyn nano_agent::channels::Channel>);
        Some(cli_ch)
    } else {
        None
    };

    if !oneshot {
        // -- GAP 7: Gateway (optional, via --gateway flag)
        if let Some(ref addr) = cli.gateway {
            let gw = Gateway::new(addr);
            gw.start();
            let ws_channel = gw.ws_channel();
            channel_manager
                .lock()
                .expect("ChannelManager lock poisoned")
                .add(ws_channel);
            // Gateway is kept alive by the channel manager holding the ws_channel Arc
        }

        // -- GAP 8: Start delivery runner
        {
            let runner =
                DeliveryRunner::new(Arc::clone(&delivery_queue), Arc::clone(&channel_manager));
            runner.start();
        }

        heartbeat_manager.start();
    }

    // -- Prompt assembler
    let mut prompt_assembler = PromptAssembler::new(&data_dir.join("prompts"));
    prompt_assembler.init_defaults();

    // -- Projector: demand-paging write gate (8K char threshold)
    let projections_dir = data_dir.join("projections");
    let projector = Projector::new(&projections_dir, 8_000);

    // -- SeedCollector: uniform system prompt contribution
    let memory_seed = Arc::new(MemorySeed::new(Arc::clone(&memory_store)));
    let mut seed_collector = SeedCollector::new();
    seed_collector.register(Arc::new(TodoSeed::new(Arc::clone(&todo_manager))));
    seed_collector.register(Arc::new(SkillSeed::new(Arc::clone(&skill_loader))));
    seed_collector.register(Arc::new(TaskSeed::new(Arc::clone(&task_manager))));
    seed_collector.register(Arc::clone(&memory_seed) as Arc<dyn nano_agent::context::Seed>);

    // -- Register self as a teammate (skip in oneshot — avoids disk writes)
    if !oneshot {
        teammate_manager
            .write()
            .expect("TeammateManager write lock poisoned")
            .spawn(&agent_name, &agent_role, "primary agent");
        // Pre-warm: list team (ensures config is written)
        let _ = teammate_manager
            .read()
            .expect("TeammateManager read lock poisoned")
            .list_all();
    }

    // -- Signals for compact, idle, and interrupt
    let compact_signal = Arc::new(CompactSignal::new());
    let idle_signal = Arc::new(AtomicBool::new(false));
    let interrupt_signal = Arc::new(AtomicBool::new(false));

    // Install Ctrl+C handler to set interrupt signal
    {
        let sig = Arc::clone(&interrupt_signal);
        ctrlc::set_handler(move || {
            sig.store(true, Ordering::Release);
        })
        .expect("Failed to install Ctrl+C handler");
    }

    // -- Build tool definitions: base (4) + extended (24)
    let mut tool_defs = tools::tool_definitions();
    tool_defs.extend(tools::extended_tool_definitions());

    // -- Build dispatch: base + extended
    let mut dispatch = tools::build_dispatch(&cwd);
    let extended = build_extended_dispatch(DispatchContext {
        config: AgentConfig {
            repo_root: cwd.to_path_buf(),
            tasks_dir: tasks_dir.clone(),
            agent_name: agent_name.clone(),
            llm_backend: backend.clone(),
        },
        services: Services {
            todo: Arc::clone(&todo_manager),
            nag: Arc::clone(&nag_policy),
            skill_loader: Arc::clone(&skill_loader),
            task_manager: Arc::clone(&task_manager),
            bg: Arc::clone(&background_manager),
            message_bus: Arc::clone(&message_bus),
            teammate_manager: Arc::clone(&teammate_manager),
            request_tracker: Arc::clone(&request_tracker),
            event_bus: Arc::clone(&event_bus),
            heartbeat_manager: Arc::clone(&heartbeat_manager),
            memory_store: Arc::clone(&memory_store),
            delivery_queue: Arc::clone(&delivery_queue),
        },
        compact_signal: Arc::clone(&compact_signal),
        idle_signal: Arc::clone(&idle_signal),
    });
    dispatch.extend(extended);

    // GAP 2: Rebuild messages from session if resuming
    let mut messages: Vec<serde_json::Value> = {
        let rebuilt = session_store.rebuild();
        if !rebuilt.is_empty() {
            if !oneshot {
                eprintln!(
                    "[session] Resumed {} messages from session {}",
                    rebuilt.len(),
                    session_id
                );
            }
            rebuilt
        } else {
            Vec::new()
        }
    };

    let mut prev_message_count = messages.len();
    let mut turn_count: usize = 0;

    // -- One-shot mode: run prompt, print response, exit
    if let Some(prompt) = oneshot_prompt {
        let projected_input = projector.project("user_input", "oneshot", &prompt);
        messages.push(serde_json::json!({
            "role": "user",
            "content": projected_input.as_str(),
        }));

        // Pre-turn pipeline (same as REPL)
        if autonomy::should_inject(&messages) {
            let identity = autonomy::make_identity_block(&agent_name, &agent_role, "nano-agent");
            messages.insert(0, identity);
        }

        // Set query for memory recall seed
        memory_seed.set_query(&prompt);

        prompt_assembler.reload();
        let ctx = build_prompt_context(
            &agent_name,
            &agent_role,
            &cwd,
            tool_defs.len(),
            &model_name,
            &agent_id,
            &session_id,
            seed_collector.render(),
        );
        let system = prompt_assembler.compose(&ctx);

        let signals = LoopSignals {
            compact_signal: Some(&compact_signal),
            transcript_dir: Some(&transcript_dir),
            idle_signal: None,
            tool_callback: None,
            interrupt_signal: Some(&interrupt_signal),
            projector: Some(&projector),
        };
        run_agent_loop(
            llm_box.as_mut(),
            &system,
            &mut messages,
            &tool_defs,
            &dispatch,
            &signals,
        );

        // Print only the final assistant response text to stdout
        if let Some(text) = extract_last_response_text(&messages) {
            print!("{}", text);
        }

        // Persist session
        session_store.append_turn(&messages[prev_message_count..]);
        transcript_store.save(&messages);
        return;
    }

    // -- Interactive REPL mode
    let history_count = std::fs::read_to_string(data_dir.join("history.txt"))
        .map(|s| s.lines().count())
        .unwrap_or(0);
    UiRenderer::splash_screen(
        &backend,
        &model_name,
        &agent_name,
        &agent_role,
        tool_defs.len(),
        &session_id,
        history_count,
    );

    let repl_start = Instant::now();
    let cli_ch = cli_channel
        .as_ref()
        .expect("CLI channel must exist in interactive mode");

    // -- Main REPL loop (reedline blocking read_line)
    loop {
        // Drain non-CLI channel messages between turns
        while let Some(msg) = channel_manager
            .lock()
            .expect("ChannelManager lock poisoned")
            .poll()
        {
            if !msg.text.is_empty() {
                UiRenderer::show_background_notification(&msg.channel, &msg.text);
            }
        }

        // Blocking read from reedline
        let input = match cli_ch.read_line() {
            Some(text) => text,
            None => continue, // Ctrl-C: ignore, show new prompt
        };

        // -- Slash commands
        match input.as_str() {
            "/quit" => break,
            "/clear" => {
                transcript_store.save(&messages);
                session_store.append_turn(&messages[prev_message_count..]);
                messages.clear();
                prev_message_count = 0;
                UiRenderer::show_warning("Transcript saved. Session finalized.");
                continue;
            }
            "/status" => {
                let tokens = memory::estimate_tokens(&messages);
                let todo_state = todo_manager
                    .read()
                    .expect("TodoManager read lock poisoned")
                    .render();
                let bg_notifs = background_manager
                    .lock()
                    .expect("BackgroundManager lock poisoned")
                    .drain_notifications();
                UiRenderer::show_status(&nano_agent::ui::StatusInfo {
                    tokens,
                    turns: messages.len(),
                    session_id: &session_id,
                    backend: &backend,
                    model: &model_name,
                    uptime: repl_start.elapsed(),
                    todo_state: &todo_state,
                    bg_count: bg_notifs.len(),
                });
                continue;
            }
            "/tasks" => {
                println!(
                    "{}\n",
                    task_manager
                        .read()
                        .expect("TaskManager read lock poisoned")
                        .list_all()
                );
                continue;
            }
            "/team" => {
                println!(
                    "{}\n",
                    teammate_manager
                        .read()
                        .expect("TeammateManager read lock poisoned")
                        .list_all()
                );
                continue;
            }
            "/events" => {
                println!("{}\n", event_bus.list_recent(20));
                continue;
            }
            "/resume" => {
                let sessions = memory::SessionStore::list_sessions(&session_dir);
                if sessions.is_empty() {
                    UiRenderer::show_warning("No sessions found.");
                } else {
                    println!("[resume] Available sessions:");
                    for (id, path) in &sessions {
                        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                        println!("  {} ({} bytes)", id, size);
                    }
                    println!("  Use: RESUME_SESSION=<id> agent  or  agent --resume <id>\n");
                }
                continue;
            }
            "/help" => {
                println!("  /quit    Exit the agent");
                println!("  /clear   Save transcript and reset conversation");
                println!("  /status  Show session status (tokens, turns, todos)");
                println!("  /tasks   List background tasks");
                println!("  /team    List teammates");
                println!("  /events  Show recent events");
                println!("  /resume  List sessions available to resume");
                println!("  /help    Show this help");
                println!();
                continue;
            }
            _ => {}
        }

        // -- Add user message (projected through demand-paging gate)
        turn_count += 1;
        let turn_key = format!("turn_{}", turn_count);
        let projected_input = projector.project("user_input", &turn_key, &input);
        messages.push(serde_json::json!({
            "role": "user",
            "content": projected_input.as_str(),
        }));

        // -- Pre-turn pipeline

        // L10: Inject identity if conversation is fresh
        if autonomy::should_inject(&messages) {
            let identity = autonomy::make_identity_block(&agent_name, &agent_role, "nano-agent");
            messages.insert(0, identity);
        }

        // L5: Token estimation and memory compaction
        let token_count = memory::estimate_tokens(&messages);
        if token_count > memory::MICRO_COMPACT_THRESHOLD {
            memory::micro_compact(&mut messages);
        }
        if token_count > memory::THRESHOLD {
            let (new_msgs, path) =
                memory::auto_compact(&messages, llm_box.as_mut(), &transcript_dir);
            messages = new_msgs;
            UiRenderer::show_compact_notice(&path.display().to_string());
        }

        // Render todo state once for nag policy
        let todo_state = todo_manager
            .read()
            .expect("TodoManager read lock poisoned")
            .render();

        // L2: Nag policy
        {
            let mut nag = nag_policy.lock().expect("NagPolicy lock poisoned");
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
            let notifs = background_manager
                .lock()
                .expect("BackgroundManager lock poisoned")
                .drain_notifications();
            if !notifs.is_empty() {
                for n in &notifs {
                    UiRenderer::show_background_notification(
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
                let projected =
                    projector.project("background", &format!("turn_{}", turn_count), &raw);
                messages.push(serde_json::json!({"role": "user", "content": projected}));
            }
        }

        // L8: Check inbox (projected)
        {
            let inbox = message_bus.read_inbox(&agent_name);
            if !inbox.is_empty() {
                let text = autonomy::format_inbox(&inbox);
                let projected = projector.project("inbox", &format!("turn_{}", turn_count), &text);
                messages.push(serde_json::json!({"role": "user", "content": projected}));
            }
        }

        // -- Heartbeat: drain cron events (projected)
        {
            let events = heartbeat_manager.drain_events();
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

        // -- Set memory recall query for this turn's seed
        memory_seed.set_query(&input);

        // -- Build dynamic system prompt via assembler with seed sections
        prompt_assembler.reload();
        let ctx = build_prompt_context(
            &agent_name,
            &agent_role,
            &cwd,
            tool_defs.len(),
            &model_name,
            &agent_id,
            &session_id,
            seed_collector.render(),
        );
        let system = prompt_assembler.compose(&ctx);

        // -- Start spinner and run the agent loop with tool callback
        let spinner = SpinnerHandle::start("thinking");

        let tool_cb = |event: ToolEvent| match event {
            ToolEvent::Start {
                ref name,
                ref input,
            } => {
                spinner.pause_and_clear();
                UiRenderer::show_tool_start(name, input);
            }
            ToolEvent::Complete {
                ref name,
                ref summary,
                duration,
            } => {
                UiRenderer::show_tool_complete(name, summary, duration);
                spinner.update_label("reasoning");
                spinner.resume();
            }
            ToolEvent::Error {
                ref name,
                ref error,
                duration,
            } => {
                UiRenderer::show_tool_error(name, error, duration);
                spinner.update_label("reasoning");
                spinner.resume();
            }
        };

        // Clear interrupt flag before each agent loop run
        interrupt_signal.store(false, Ordering::Release);

        let signals = LoopSignals {
            compact_signal: Some(&compact_signal),
            transcript_dir: Some(&transcript_dir),
            idle_signal: None,
            tool_callback: Some(&tool_cb),
            interrupt_signal: Some(&interrupt_signal),
            projector: Some(&projector),
        };
        let _calls = run_agent_loop(
            llm_box.as_mut(),
            &system,
            &mut messages,
            &tool_defs,
            &dispatch,
            &signals,
        );

        // Stop spinner (Drop joins the thread)
        drop(spinner);

        // Show interruption notice if user pressed Ctrl+C
        if interrupt_signal.load(Ordering::Acquire) {
            UiRenderer::show_warning("Interrupted.");
            interrupt_signal.store(false, Ordering::Release);
        }

        // -- GAP 2: Persist new messages to session
        if messages.len() > prev_message_count {
            session_store.append_turn(&messages[prev_message_count..]);
            prev_message_count = messages.len();
        }

        // -- Display the response with formatting
        if let Some(text) = extract_last_response_text(&messages) {
            UiRenderer::show_response(&text);
        }

        // -- Update prompt state for next turn
        {
            let token_count = memory::estimate_tokens(&messages);
            let mut ps = prompt_state.lock().expect("PromptState lock poisoned");
            ps.turn += 1;
            ps.tokens = token_count;
        }

        // Sync history to disk
        cli_ch.sync_history();
    }

    // -- Save transcript on exit
    if !messages.is_empty() {
        transcript_store.save(&messages);
        session_store.append_turn(&messages[prev_message_count..]);
        UiRenderer::show_warning("Transcript and session saved.");
    }
}

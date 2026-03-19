use clap::Parser;
use nano_agent::anthropic::AnthropicLlm;
use nano_agent::channels::Channel;
use nano_agent::channels::{ChannelManager, CliChannel};
use nano_agent::cli::Cli;
use nano_agent::concurrency::BackgroundManager;
use nano_agent::context::{MemorySeed, Projector, SeedCollector, SkillSeed, TaskSeed, TodoSeed};
use nano_agent::core_loop::run_agent_loop;
use nano_agent::delivery::{DeliveryQueue, DeliveryRunner};
use nano_agent::dispatch::{auth_prefix_for, build_extended_registry, wrap_llm};
use nano_agent::gateway::Gateway;
use nano_agent::gateway::WebSocketChannel;
use nano_agent::handler::{
    AgentContext, AgentIdentity, AgentServices, AgentSignals, HandlerRegistry,
};
use nano_agent::heartbeat::HeartbeatManager;
use nano_agent::isolation::EventBus;
use nano_agent::knowledge::SkillLoader;
use nano_agent::memory;
use nano_agent::memory_store::MemoryStore;
use nano_agent::openai::OpenAiLlm;
use nano_agent::pipeline::{build_pre_turn_context, PreTurnUi};
use nano_agent::planning::{NagPolicy, TodoManager};
use nano_agent::prompt::{build_prompt_context, extract_last_response_text, PromptAssembler};
use nano_agent::protocols::RequestTracker;
use nano_agent::tasks::TaskManager;
use nano_agent::teams::{MessageBus, TeammateManager};
use nano_agent::tools;
use nano_agent::tui;
use nano_agent::tui::events::LogLevel as UiLogLevel;
use nano_agent::tui::theme::Theme as TuiTheme;
use nano_agent::types::*;
use nano_agent::ui::{NanoPrompt, PromptState, SpinnerHandle, UiRenderer};
use nano_agent::ws_jsonrpc;
use nano_agent::{agent_core, ws_jsonrpc::JsonRpcNotification};
use std::collections::HashMap;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SESSION_EVICTION_DAYS: u64 = 30;
const MEMORY_EVICTION_DAYS: u64 = 90;
const PROJECTOR_CHAR_THRESHOLD: usize = 8_000;
const NAG_POLICY_INTERVAL: usize = 3;

const VALID_BACKENDS: &[&str] = &["anthropic", "openai"];
const RECENT_EVENTS_COUNT: usize = 20;

struct PreTurnUiRepl;
impl PreTurnUi for PreTurnUiRepl {
    fn compact_notice(&self, path: &str) {
        UiRenderer::show_compact_notice(path);
    }
    fn background_notification(&self, source: &str, message: &str) {
        UiRenderer::show_background_notification(source, message);
    }
}

struct PreTurnUiNoop;
impl PreTurnUi for PreTurnUiNoop {
    fn compact_notice(&self, _path: &str) {}
    fn background_notification(&self, _source: &str, _message: &str) {}
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
    backend: String,
    agent_name: String,
    agent_role: String,
    skills_dir: Option<PathBuf>,
    resume_session: Option<String>,
}

impl Config {
    fn from_env(cli: &Cli) -> Self {
        let raw_backend = std::env::var("LLM_BACKEND").unwrap_or_else(|_| "anthropic".into());
        let backend = if VALID_BACKENDS.contains(&raw_backend.as_str()) {
            raw_backend
        } else {
            eprintln!(
                "warning: unknown LLM_BACKEND {:?}, falling back to \"anthropic\" (valid: {:?})",
                raw_backend, VALID_BACKENDS
            );
            "anthropic".into()
        };
        let agent_name = std::env::var("AGENT_NAME").unwrap_or_else(|_| "lead".into());
        let agent_role = std::env::var("AGENT_ROLE").unwrap_or_else(|_| "developer".into());
        let skills_dir = std::env::var("SKILLS_DIR").map(PathBuf::from).ok();
        let resume_session = std::env::var("RESUME_SESSION")
            .ok()
            .or_else(|| cli.resume.clone());
        Self {
            backend,
            agent_name,
            agent_role,
            skills_dir,
            resume_session,
        }
    }
}

// ---------------------------------------------------------------------------
// Slash command handling
// ---------------------------------------------------------------------------

enum SlashResult {
    Handled,
    Quit,
    NotSlash,
}

#[allow(clippy::too_many_arguments)]
fn handle_slash_command(
    input: &str,
    messages: &mut Vec<serde_json::Value>,
    prev_message_count: &mut usize,
    transcript_store: &memory::TranscriptStore,
    ctx: &AgentContext,
    session_dir: &std::path::Path,
    repl_start: Instant,
    model_name: &str,
) -> SlashResult {
    let identity = &ctx.identity;
    match input {
        "/quit" => SlashResult::Quit,
        "/clear" => {
            transcript_store.save(messages);
            ctx.services
                .sessions
                .append_turn(&messages[*prev_message_count..]);
            messages.clear();
            *prev_message_count = 0;
            UiRenderer::show_warning("Transcript saved. Session finalized.");
            SlashResult::Handled
        }
        "/status" => {
            let tokens = memory::estimate_tokens(messages);
            let todo_state = ctx
                .services
                .todo
                .read()
                .expect("TodoManager read lock poisoned")
                .render();
            let bg_notifs = ctx
                .services
                .bg
                .lock()
                .expect("BackgroundManager lock poisoned")
                .drain_notifications();
            let teammate_state = ctx
                .services
                .teammate_manager
                .read()
                .expect("TeammateManager read lock poisoned")
                .list_all();
            UiRenderer::show_status(&nano_agent::ui::StatusInfo {
                tokens,
                turns: messages.len(),
                session_id: &identity.session_id,
                backend: &ctx.llm_backend,
                model: model_name,
                uptime: repl_start.elapsed(),
                todo_state: &todo_state,
                bg_count: bg_notifs.len(),
                teammate_state: &teammate_state,
            });
            SlashResult::Handled
        }
        "/tasks" => {
            println!(
                "{}\n",
                ctx.services
                    .task_manager
                    .read()
                    .expect("TaskManager read lock poisoned")
                    .list_all()
            );
            SlashResult::Handled
        }
        "/team" => {
            println!(
                "{}\n",
                ctx.services
                    .teammate_manager
                    .read()
                    .expect("TeammateManager read lock poisoned")
                    .list_all()
            );
            SlashResult::Handled
        }
        "/events" => {
            println!(
                "{}\n",
                ctx.services.event_bus.list_recent(RECENT_EVENTS_COUNT)
            );
            SlashResult::Handled
        }
        "/resume" => {
            let sessions = memory::SessionStore::list_sessions(session_dir);
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
            SlashResult::Handled
        }
        "/help" => {
            println!("  /quit    Exit the agent");
            println!("  /clear   Save transcript and reset conversation");
            println!("  /status  Show session status (tokens, turns, todos)");
            println!("  /tasks   List background tasks");
            println!("  /team    List teammates (status: working, idle)");
            println!("  /events  Show recent events");
            println!("  /resume  List sessions available to resume");
            println!("  /help    Show this help");
            println!();
            SlashResult::Handled
        }
        _ => SlashResult::NotSlash,
    }
}

// ---------------------------------------------------------------------------
// Oneshot mode
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_oneshot(
    prompt: &str,
    messages: &mut Vec<serde_json::Value>,
    ctx: &AgentContext,
    llm: &mut dyn Llm,
    projector: &Projector,
    memory_seed: &MemorySeed,
    prompt_assembler: &mut PromptAssembler,
    seed_collector: &Arc<Mutex<SeedCollector>>,
    tool_defs: &[serde_json::Value],
    model_name: &str,
    registry: &HandlerRegistry,
    prev_message_count: usize,
    transcript_store: &memory::TranscriptStore,
) {
    let projected_input = projector.project("user_input", "oneshot", prompt);
    messages.push(serde_json::json!({
        "role": "user",
        "content": projected_input.as_str(),
    }));

    // Pre-turn pipeline (same stages as REPL)
    let transcript_dir = ctx
        .transcript_dir
        .as_deref()
        .expect("transcript_dir required for oneshot");
    build_pre_turn_context(
        messages,
        ctx,
        llm,
        transcript_dir,
        projector,
        1,
        memory_seed,
        prompt,
        &PreTurnUiNoop,
    );

    prompt_assembler.reload();
    let prompt_ctx = build_prompt_context(
        &ctx.identity.name,
        &ctx.identity.role,
        &ctx.cwd,
        tool_defs.len(),
        model_name,
        &ctx.identity.id,
        &ctx.identity.session_id,
        seed_collector
            .lock()
            .expect("SeedCollector lock poisoned")
            .render(),
    );
    let system = prompt_assembler.compose(&prompt_ctx);

    run_agent_loop(llm, &system, messages, tool_defs, registry, ctx);

    // Print only the final assistant response text to stdout (trailing newline for clean shell)
    if let Some(text) = extract_last_response_text(messages) {
        println!("{}", text.trim_end());
    }

    // Persist session
    ctx.services
        .sessions
        .append_turn(&messages[prev_message_count..]);
    transcript_store.save(messages);
}

// ---------------------------------------------------------------------------
// Tool callback builder
// ---------------------------------------------------------------------------

fn build_tool_callback(spinner: &Arc<SpinnerHandle>) -> Arc<dyn Fn(ToolEvent) + Send + Sync> {
    let spinner_ref = Arc::clone(spinner);
    Arc::new(move |event: ToolEvent| match event {
        ToolEvent::Start {
            ref name,
            ref input,
        } => {
            spinner_ref.pause_and_clear();
            UiRenderer::show_tool_start(name, input);
            // Delegation visibility: show who work was delegated to
            if name == "subagent" {
                if let Some(preview) = input.get("prompt").and_then(|v| v.as_str()) {
                    UiRenderer::show_delegation_to_subagent(preview);
                }
            } else if name == "send_message" {
                if let (Some(to), Some(content)) = (
                    input.get("to").and_then(|v| v.as_str()),
                    input.get("content").and_then(|v| v.as_str()),
                ) {
                    UiRenderer::show_delegation_to_teammate(to, content);
                }
            } else if name == "broadcast_message" {
                if let Some(content) = input.get("content").and_then(|v| v.as_str()) {
                    UiRenderer::show_delegation_broadcast(content);
                }
            }
        }
        ToolEvent::Complete {
            ref name,
            ref summary,
            duration,
        } => {
            UiRenderer::show_tool_complete(name, summary, duration);
            spinner_ref.update_label("reasoning");
            spinner_ref.resume();
        }
        ToolEvent::Error {
            ref name,
            ref error,
            duration,
        } => {
            UiRenderer::show_tool_error(name, error, duration);
            spinner_ref.update_label("reasoning");
            spinner_ref.resume();
        }
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let config = Config::from_env(&cli);

    let cwd = std::env::current_dir().expect("Failed to get current directory");
    let data_dir = cwd.join(".nano-agent");
    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        eprintln!(
            "warning: failed to create data directory {:?}: {}",
            data_dir, e
        );
    }

    // -- Detect piped stdin and build oneshot prompt
    let piped_input = if !std::io::stdin().is_terminal() {
        let mut buf = String::new();
        if let Err(e) = std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf) {
            eprintln!("warning: failed to read from stdin: {}", e);
        }
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
    let interactive_tty = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let prefer_tui = interactive_tty && !oneshot && !cli.repl;
    let mut tui_mode = cli.tui || prefer_tui;
    if tui_mode && !interactive_tty {
        eprintln!("warning: --tui requires a TTY; falling back to non-interactive behavior");
        tui_mode = false;
    }
    if !interactive_tty && !oneshot && !cli.repl && cli.gateway.is_none() {
        eprintln!("error: interactive mode requires a TTY. Provide a prompt (one-shot), use --repl, or use --gateway for headless.");
        std::process::exit(2);
    }

    // If we're going to run the full-screen TUI, create the UI event channel early so
    // LLM wrappers (resilience/context-guard) can log into it from the start.
    let mut tui_channel: Option<(
        tui::events::UiEventSender,
        mpsc::Receiver<tui::events::UiEvent>,
    )> = if tui_mode {
        let (tx, rx) = mpsc::channel::<tui::events::UiEvent>();
        Some((tui::events::UiEventSender::new(tx), rx))
    } else {
        None
    };

    let llm_log_sink: nano_agent::resilience::LlmLogSink = tui_channel.as_ref().map(|(tx, _)| {
        let tx = tx.clone();
        Arc::new(
            move |source: &'static str, level: &'static str, msg: &str| {
                let lvl = match level {
                    "debug" => UiLogLevel::Debug,
                    "warn" => UiLogLevel::Warn,
                    "error" => UiLogLevel::Error,
                    _ => UiLogLevel::Info,
                };
                let _ = tx.send(tui::events::UiEvent::LogLine {
                    source: source.to_string(),
                    level: lvl,
                    message: msg.to_string(),
                });
            },
        ) as Arc<dyn Fn(&'static str, &'static str, &str) + Send + Sync>
    });

    // -- LLM backend (wrapped in resilience + ContextGuard layers)
    let transcript_dir = data_dir.join("transcripts");
    let (mut llm_box, model_name): (Box<dyn Llm>, String) = {
        let use_mock_llm = cli.gateway.is_some()
            && !tui_mode
            && !cli.repl
            && std::env::var("MOCK_LLM").as_deref() == Ok("1");
        if use_mock_llm {
            let mut mock = nano_agent::mock::MockLLM::new();
            mock.queue("end_turn", vec![nano_agent::mock::make_text_block("OK")]);
            (Box::new(mock) as Box<dyn Llm>, "mock".to_string())
        } else {
            let (inner, model) = match config.backend.as_str() {
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
                wrap_llm(
                    inner,
                    auth_prefix_for(&config.backend),
                    &transcript_dir,
                    llm_log_sink,
                ),
                model,
            )
        }
    };

    // -- Data directories
    let tasks_dir = data_dir.join("tasks");
    let session_dir = data_dir.join("sessions");
    let inbox_dir = data_dir.join("inbox");
    let team_dir = data_dir.join("team");
    let events_path = data_dir.join("events.jsonl");
    let delivery_dir = data_dir.join("delivery");
    let memories_dir = data_dir.join("memories");
    // -- Destructure config (consumes all fields, prevents partial-move footguns)
    let Config {
        backend,
        agent_name,
        agent_role,
        skills_dir: skills_dir_override,
        resume_session,
    } = config;
    let skills_dir = skills_dir_override.unwrap_or_else(|| data_dir.join("skills"));
    let agent_id = uuid::Uuid::new_v4().to_string();

    // -- GAP 2: Session management
    let session_id = resume_session.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let session_store = memory::SessionStore::new(&session_dir, &session_id);

    // -- Evict old data files at startup (skip in oneshot to avoid unnecessary I/O)
    if !oneshot {
        memory::SessionStore::evict_old_sessions(&session_dir, SESSION_EVICTION_DAYS);
    }

    // -- Initialize all managers
    let task_manager = Arc::new(RwLock::new(TaskManager::new(&tasks_dir)));
    let event_bus = Arc::new(EventBus::new(&events_path));
    // Subscriber is installed later once we know the UI mode.
    let skill_loader = Arc::new(SkillLoader::new(&skills_dir));
    let transcript_store = memory::TranscriptStore::new(&transcript_dir);
    let background_manager = Arc::new(Mutex::new(BackgroundManager::new()));
    let todo_manager = Arc::new(RwLock::new(TodoManager::new()));
    let nag_policy = Arc::new(Mutex::new(NagPolicy::new(NAG_POLICY_INTERVAL)));
    let message_bus = Arc::new(MessageBus::new(&inbox_dir));
    let teammate_manager = Arc::new(RwLock::new(TeammateManager::new(&team_dir)));
    let request_tracker = Arc::new(RequestTracker::new());

    // -- GAP 5: Memory store with TF-IDF search
    let memory_store_inner =
        MemoryStore::new(&data_dir.join("prompts").join("MEMORY.md"), &memories_dir);
    if !oneshot {
        memory_store_inner.evict_old(MEMORY_EVICTION_DAYS);
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
    let cli_channel: Option<Arc<CliChannel>> = if !oneshot && !tui_mode && cli.gateway.is_none() {
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

    let mut ws_channel_for_gateway: Option<Arc<WebSocketChannel>> = None;
    if !oneshot {
        // -- GAP 7: Gateway (optional, via --gateway flag)
        if let Some(ref addr) = cli.gateway {
            let gw = Gateway::new(addr);
            gw.start();
            let ws_channel = gw.ws_channel();
            ws_channel_for_gateway = Some(Arc::clone(&ws_channel));
            channel_manager
                .lock()
                .expect("ChannelManager lock poisoned")
                .add(ws_channel);
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
    let projector = Arc::new(Projector::new(&projections_dir, PROJECTOR_CHAR_THRESHOLD));

    // -- SeedCollector: uniform system prompt contribution
    let memory_seed = Arc::new(MemorySeed::new(Arc::clone(&memory_store)));
    let seed_collector = Arc::new(Mutex::new(SeedCollector::new()));
    {
        let mut sc = seed_collector.lock().expect("SeedCollector lock poisoned");
        sc.register(Arc::new(TodoSeed::new(Arc::clone(&todo_manager))));
        sc.register(Arc::new(SkillSeed::new(Arc::clone(&skill_loader))));
        sc.register(Arc::new(TaskSeed::new(Arc::clone(&task_manager))));
        sc.register(Arc::clone(&memory_seed) as Arc<dyn nano_agent::context::Seed>);
    }

    // -- Register self as a teammate (skip in oneshot — avoids disk writes)
    if !oneshot {
        teammate_manager
            .write()
            .expect("TeammateManager write lock poisoned")
            .spawn(&agent_name, &agent_role, "primary agent");
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

    // -- Build tool definitions: base (7) + extended (28)
    let mut tool_defs = tools::tool_definitions();
    tool_defs.extend(tools::extended_tool_definitions());

    // -- Build AgentContext
    let ctx = AgentContext {
        identity: AgentIdentity {
            name: agent_name.clone(),
            role: agent_role.clone(),
            id: agent_id.clone(),
            session_id: session_id.clone(),
        },
        tasks_dir: tasks_dir.clone(),
        llm_backend: backend.clone(),
        cwd: cwd.to_path_buf(),
        services: AgentServices {
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
            channels: Arc::clone(&channel_manager),
            sessions: Arc::new(session_store),
        },
        signals: AgentSignals {
            compact: Arc::clone(&compact_signal),
            idle: Arc::clone(&idle_signal),
            interrupt: Some(Arc::clone(&interrupt_signal)),
        },
        projector: Some(Arc::clone(&projector)),
        transcript_dir: Some(transcript_dir.clone()),
        tool_callback: None, // set per-loop-iteration in REPL
        subagent_progress: if oneshot {
            None
        } else {
            Some(Arc::new(|msg: &str| {
                UiRenderer::show_subagent_progress(msg)
            }))
        },
    };

    // -- Build HandlerRegistry: base + extended
    let mut registry = tools::build_registry(&cwd);
    registry.extend(build_extended_registry(&ctx));
    let registry = Arc::new(registry);

    // GAP 2: Rebuild messages from session if resuming
    let mut messages: Vec<serde_json::Value> = {
        let rebuilt = ctx.services.sessions.rebuild();
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
        run_oneshot(
            &prompt,
            &mut messages,
            &ctx,
            llm_box.as_mut(),
            &projector,
            &memory_seed,
            &mut prompt_assembler,
            &seed_collector,
            &tool_defs,
            &model_name,
            registry.as_ref(),
            prev_message_count,
            &transcript_store,
        );
        return;
    }

    // -- Headless gateway mode: WebSocket JSON-RPC control plane (no REPL/TUI)
    if cli.gateway.is_some() && !tui_mode && !cli.repl {
        let ws_channel = ws_channel_for_gateway
            .as_ref()
            .expect("ws_channel must exist when --gateway is set");

        let core = agent_core::AgentCore::start(
            ctx.clone(),
            llm_box,
            Arc::clone(&registry),
            Arc::new(tool_defs),
            model_name.clone(),
            data_dir.join("prompts"),
            transcript_dir.clone(),
            Arc::clone(&projector),
            Arc::clone(&memory_seed),
            Arc::clone(&seed_collector),
        );

        let pending: Arc<Mutex<HashMap<String, (serde_json::Value, String)>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Inbound: ws -> JSON-RPC -> core commands
        {
            let ws_in = Arc::clone(ws_channel);
            let core_in = core.clone();
            let pending_in = Arc::clone(&pending);
            std::thread::spawn(move || loop {
                if let Some(msg) = ws_in.recv() {
                    let req = match ws_jsonrpc::parse_request(&msg.text) {
                        Ok(r) => r,
                        Err(e) => {
                            let _ = ws_in.send(
                                &msg.peer_id,
                                &ws_jsonrpc::err(serde_json::Value::Null, -32700, e),
                            );
                            continue;
                        }
                    };
                    if req.jsonrpc != "2.0" {
                        let _ = ws_in.send(
                            &msg.peer_id,
                            &ws_jsonrpc::err(req.id, -32600, "jsonrpc must be '2.0'"),
                        );
                        continue;
                    }

                    let request_id = match &req.id {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };

                    match req.method.as_str() {
                        "agent.run_turn" => {
                            let id_value = req.id.clone();
                            let prompt = req
                                .params
                                .get("prompt")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());
                            let include_bus = req
                                .params
                                .get("include_bus_events")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            let Some(prompt) = prompt else {
                                let _ = ws_in.send(
                                    &msg.peer_id,
                                    &ws_jsonrpc::err(
                                        req.id,
                                        -32602,
                                        "params.prompt (string) required",
                                    ),
                                );
                                continue;
                            };
                            pending_in
                                .lock()
                                .expect("pending lock poisoned")
                                .insert(request_id.clone(), (id_value, msg.peer_id.clone()));
                            if let Err(e) = agent_core::AgentCore::enqueue_turn(
                                &core_in,
                                request_id.clone(),
                                Some(msg.peer_id.clone()),
                                prompt,
                                include_bus,
                            ) {
                                let _ =
                                    ws_in.send(&msg.peer_id, &ws_jsonrpc::err(req.id, -32000, e));
                            }
                        }
                        "agent.status" => {
                            pending_in
                                .lock()
                                .expect("pending lock poisoned")
                                .insert(request_id.clone(), (req.id.clone(), msg.peer_id.clone()));
                            let _ = core_in.cmd_tx.send(agent_core::AgentCommand::Status {
                                request_id,
                                peer_id: Some(msg.peer_id.clone()),
                            });
                        }
                        "agent.interrupt" => {
                            let _ = core_in
                                .cmd_tx
                                .send(agent_core::AgentCommand::Interrupt { request_id: None });
                            let _ = ws_in.send(
                                &msg.peer_id,
                                &ws_jsonrpc::ok(req.id, serde_json::json!({"ok": true})),
                            );
                        }
                        _ => {
                            let _ = ws_in.send(
                                &msg.peer_id,
                                &ws_jsonrpc::err(req.id, -32601, "method not found"),
                            );
                        }
                    }
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            });
        }

        // Outbound: core events -> JSON-RPC notifications/responses
        {
            let ws_out = Arc::clone(ws_channel);
            let pending_out = Arc::clone(&pending);
            let event_rx = Arc::clone(&core.event_rx);
            std::thread::spawn(move || loop {
                let env = {
                    let rx = event_rx.lock().expect("event_rx lock poisoned");
                    rx.recv()
                };
                let Ok(env) = env else { break };

                let peer = match env.peer_id.clone() {
                    Some(p) => p,
                    None => continue,
                };

                // Responses (Status, TurnFinished, Error) are tied to pending request ids.
                match &env.event {
                    agent_core::AgentEvent::Status { .. } => {
                        if let Some((id_value, _peer0)) = pending_out
                            .lock()
                            .expect("pending lock poisoned")
                            .remove(&env.request_id)
                        {
                            let _ = ws_out.send(&peer, &ws_jsonrpc::ok(id_value, &env.event));
                        }
                        continue;
                    }
                    agent_core::AgentEvent::TurnFinished { assistant_text, .. } => {
                        if let Some((id_value, _peer0)) = pending_out
                            .lock()
                            .expect("pending lock poisoned")
                            .remove(&env.request_id)
                        {
                            let _ = ws_out.send(
                                &peer,
                                &ws_jsonrpc::ok(
                                    id_value,
                                    serde_json::json!({ "assistant_text": assistant_text }),
                                ),
                            );
                        }
                        // Also emit as event notification for streaming clients.
                    }
                    agent_core::AgentEvent::Error { message, .. } => {
                        if let Some((id_value, _peer0)) = pending_out
                            .lock()
                            .expect("pending lock poisoned")
                            .remove(&env.request_id)
                        {
                            let _ = ws_out.send(&peer, &ws_jsonrpc::err(id_value, -32001, message));
                            continue;
                        }
                    }
                    _ => {}
                }

                let notif = JsonRpcNotification {
                    jsonrpc: "2.0",
                    method: "agent.event",
                    params: serde_json::json!({
                        "request_id": env.request_id,
                        "event": env.event,
                    }),
                };
                if let Ok(text) = serde_json::to_string(&notif) {
                    let _ = ws_out.send(&peer, &text);
                }
            });
        }

        // Block forever; Ctrl+C will terminate the process.
        loop {
            std::thread::park_timeout(std::time::Duration::from_secs(3600));
        }
    }

    // -- UI wiring for interactive modes (REPL or TUI)
    {
        let bus = Arc::clone(&event_bus);
        if tui_mode {
            // TUI subscribes later with its own event channel (see below).
        } else {
            bus.subscribe(Arc::new(
                move |record: nano_agent::isolation::EventRecord| {
                    UiRenderer::show_bus_event(&record.event, &record.data);
                },
            ));
        }
    }

    // -- Full-screen TUI mode (default for interactive tty)
    if tui_mode {
        let theme = match cli.theme.as_str() {
            "bold" | "BOLD" => TuiTheme::Bold,
            "hacker" | "HACKER" => TuiTheme::Hacker,
            _ => TuiTheme::Bold,
        };

        let messages_arc = Arc::new(Mutex::new(messages));
        let llm_arc = Arc::new(Mutex::new(llm_box));
        let registry_arc = Arc::clone(&registry);
        let tool_defs_arc = Arc::new(tool_defs);

        let seed_collector = Arc::clone(&seed_collector);

        let turn_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prev_message_count_arc = Arc::new(Mutex::new(prev_message_count));

        // UI event bridge (tool/bus/background/agent-finished + resilience/context-guard logs)
        let (ui_tx, ui_rx) = tui_channel
            .take()
            .expect("tui_channel must be initialized when tui_mode is true");

        // Subscribe bus → TUI
        {
            let ui_tx_for_bus = ui_tx.clone();
            event_bus.subscribe(Arc::new(
                move |record: nano_agent::isolation::EventRecord| {
                    let _ = ui_tx_for_bus.send(tui::events::UiEvent::BusEvent {
                        event: record.event,
                        data: record.data,
                    });
                },
            ));
        }

        struct PreTurnUiTui {
            tx: tui::events::UiEventSender,
        }
        impl PreTurnUi for PreTurnUiTui {
            fn compact_notice(&self, path: &str) {
                let _ = self.tx.send(tui::events::UiEvent::Warning(format!(
                    "Compacted. Transcript: {}",
                    path
                )));
            }
            fn background_notification(&self, source: &str, message: &str) {
                let _ = self.tx.send(tui::events::UiEvent::Background {
                    source: source.to_string(),
                    message: message.to_string(),
                });
            }
        }
        let preturn_ui = Arc::new(PreTurnUiTui { tx: ui_tx.clone() });

        let prompts_dir = data_dir.join("prompts");
        let transcript_dir_for_tui = transcript_dir.clone();
        let projector_for_tui = Arc::clone(&projector);
        let memory_seed_for_tui = Arc::clone(&memory_seed);

        let model_name_for_tui = model_name.clone();
        let agent_id_for_tui = agent_id.clone();
        let session_id_for_tui = session_id.clone();
        let agent_name_for_tui = agent_name.clone();
        let agent_role_for_tui = agent_role.clone();

        let ctx_for_launch = ctx.clone();
        let messages_for_launch = Arc::clone(&messages_arc);
        let ui_tx_for_launch = ui_tx.clone();

        let launch_agent_turn =
            move |prompt: String, tool_cb: Arc<dyn Fn(ToolEvent) + Send + Sync>| {
                let ctx = ctx_for_launch.clone();
                let messages_arc = Arc::clone(&messages_for_launch);
                let llm_arc = Arc::clone(&llm_arc);
                let registry = Arc::clone(&registry_arc);
                let tool_defs = Arc::clone(&tool_defs_arc);
                let turn_counter = Arc::clone(&turn_counter);
                let prev_message_count_arc = Arc::clone(&prev_message_count_arc);
                let seed_collector = Arc::clone(&seed_collector);

                let projector = Arc::clone(&projector_for_tui);
                let memory_seed = Arc::clone(&memory_seed_for_tui);
                let prompts_dir = prompts_dir.clone();
                let transcript_dir = transcript_dir_for_tui.clone();

                let ui_tx = ui_tx_for_launch.clone();
                let preturn_ui = Arc::clone(&preturn_ui);

                let model_name = model_name_for_tui.clone();
                let agent_id = agent_id_for_tui.clone();
                let session_id = session_id_for_tui.clone();
                let agent_name = agent_name_for_tui.clone();
                let agent_role = agent_role_for_tui.clone();

                interrupt_signal.store(false, Ordering::Release);

                std::thread::spawn(move || {
                    let turn = turn_counter.fetch_add(1, Ordering::AcqRel) + 1;
                    let turn_key = format!("turn_{}", turn);

                    let mut messages = messages_arc.lock().expect("messages lock poisoned");
                    let projected_input = projector.project("user_input", &turn_key, &prompt);
                    messages.push(
                        serde_json::json!({"role":"user","content": projected_input.as_str()}),
                    );

                    // Pre-turn pipeline + prompt build needs LLM (for compaction) and dynamic seeds.
                    let mut llm_guard = llm_arc.lock().expect("llm lock poisoned");
                    let token_count = build_pre_turn_context(
                        &mut messages,
                        &ctx,
                        llm_guard.as_mut(),
                        &transcript_dir,
                        &projector,
                        turn,
                        &memory_seed,
                        &prompt,
                        preturn_ui.as_ref(),
                    );

                    let seed_render = seed_collector
                        .lock()
                        .expect("SeedCollector lock poisoned")
                        .render();
                    let mut prompt_assembler = PromptAssembler::new(&prompts_dir);
                    prompt_assembler.reload();
                    let prompt_ctx = build_prompt_context(
                        &agent_name,
                        &agent_role,
                        &ctx.cwd,
                        tool_defs.len(),
                        &model_name,
                        &agent_id,
                        &session_id,
                        seed_render,
                    );
                    let system = prompt_assembler.compose(&prompt_ctx);

                    let ui_tx_progress = ui_tx.clone();
                    let loop_ctx = AgentContext {
                        tool_callback: Some(tool_cb),
                        subagent_progress: Some(Arc::new(move |msg: &str| {
                            let _ = ui_tx_progress.send(tui::events::UiEvent::SubagentProgress {
                                message: msg.to_string(),
                            });
                        })),
                        ..ctx.clone()
                    };
                    run_agent_loop(
                        llm_guard.as_mut(),
                        &system,
                        &mut messages,
                        &tool_defs,
                        registry.as_ref(),
                        &loop_ctx,
                    );

                    // Persist new messages
                    {
                        let mut prev = prev_message_count_arc
                            .lock()
                            .expect("prev_count lock poisoned");
                        if messages.len() > *prev {
                            ctx.services.sessions.append_turn(&messages[*prev..]);
                            *prev = messages.len();
                        }
                    }

                    if let Some(text) = extract_last_response_text(&messages) {
                        let _ = ui_tx.send(tui::events::UiEvent::AgentFinished {
                            assistant_text: text,
                        });
                    } else {
                        let _ = ui_tx.send(tui::events::UiEvent::Error(
                            "No assistant response found.".to_string(),
                        ));
                    }

                    let _ = token_count;
                });
            };

        let opts = tui::TuiOptions {
            no_color: cli.no_color,
            theme,
        };
        let ui_tx_for_tui = ui_tx.clone();
        let _ = tui::run_tui(
            ctx,
            Arc::clone(&messages_arc),
            model_name.clone(),
            opts,
            ui_tx_for_tui,
            ui_rx,
            launch_agent_turn,
        );
        return;
    }

    // -- Interactive REPL mode (legacy)
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
        match handle_slash_command(
            &input,
            &mut messages,
            &mut prev_message_count,
            &transcript_store,
            &ctx,
            &session_dir,
            repl_start,
            &model_name,
        ) {
            SlashResult::Quit => break,
            SlashResult::Handled => continue,
            SlashResult::NotSlash => {}
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
        let token_count = build_pre_turn_context(
            &mut messages,
            &ctx,
            llm_box.as_mut(),
            &transcript_dir,
            &projector,
            turn_count,
            &memory_seed,
            &input,
            &PreTurnUiRepl,
        );

        // -- Build dynamic system prompt via assembler with seed sections
        prompt_assembler.reload();
        let prompt_ctx = build_prompt_context(
            &agent_name,
            &agent_role,
            &cwd,
            tool_defs.len(),
            &model_name,
            &agent_id,
            &session_id,
            seed_collector
                .lock()
                .expect("SeedCollector lock poisoned")
                .render(),
        );
        let system = prompt_assembler.compose(&prompt_ctx);

        // -- Start spinner and run the agent loop with tool callback
        let spinner = Arc::new(SpinnerHandle::start("thinking"));
        let tool_cb = build_tool_callback(&spinner);

        // Clear interrupt flag before each agent loop run
        interrupt_signal.store(false, Ordering::Release);

        // Create per-iteration context with callback
        let loop_ctx = AgentContext {
            tool_callback: Some(tool_cb),
            ..ctx.clone()
        };
        let _calls = run_agent_loop(
            llm_box.as_mut(),
            &system,
            &mut messages,
            &tool_defs,
            registry.as_ref(),
            &loop_ctx,
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
            ctx.services
                .sessions
                .append_turn(&messages[prev_message_count..]);
            prev_message_count = messages.len();
        }

        // -- Display the response with formatting
        if let Some(text) = extract_last_response_text(&messages) {
            UiRenderer::show_response(&text);
        }

        // -- Update prompt state for next turn (reuse token_count from pre-turn)
        {
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
        ctx.services
            .sessions
            .append_turn(&messages[prev_message_count..]);
        UiRenderer::show_warning("Transcript and session saved.");
    }
}

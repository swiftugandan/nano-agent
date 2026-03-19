use crate::context::{MemorySeed, Projector, SeedCollector};
use crate::core_loop::run_agent_loop;
use crate::handler::{AgentContext, HandlerRegistry};
use crate::isolation::EventRecord;
use crate::pipeline::{build_pre_turn_context, PreTurnUi};
use crate::prompt::{build_prompt_context, extract_last_response_text, PromptAssembler};
use crate::types::{ToolEvent, Llm};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::atomic::Ordering;
use std::sync::mpsc;

#[derive(Debug, Clone)]
pub enum AgentCommand {
    RunTurn {
        request_id: String,
        peer_id: Option<String>,
        prompt: String,
        include_bus_events: bool,
    },
    Interrupt {
        request_id: Option<String>,
    },
    Status {
        request_id: String,
        peer_id: Option<String>,
    },
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentEvent {
    #[serde(rename = "turn_started")]
    TurnStarted {
        request_id: String,
    },

    #[serde(rename = "tool_start")]
    ToolStart {
        request_id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_complete")]
    ToolComplete {
        request_id: String,
        name: String,
        summary: String,
        duration_ms: u64,
    },
    #[serde(rename = "tool_error")]
    ToolError {
        request_id: String,
        name: String,
        error: String,
        duration_ms: u64,
    },

    #[serde(rename = "bus_event")]
    BusEvent {
        request_id: String,
        event: String,
        data: serde_json::Value,
    },

    #[serde(rename = "warning")]
    Warning {
        request_id: String,
        message: String,
    },
    #[serde(rename = "error")]
    Error {
        request_id: String,
        message: String,
    },

    #[serde(rename = "turn_finished")]
    TurnFinished {
        request_id: String,
        assistant_text: String,
    },

    #[serde(rename = "status")]
    Status {
        request_id: String,
        busy: bool,
        queued: usize,
        agent_name: String,
        agent_role: String,
        session_id: String,
    },
}

#[derive(Debug, Clone)]
pub struct AgentEventEnvelope {
    pub peer_id: Option<String>,
    pub request_id: String,
    pub event: AgentEvent,
}

#[derive(Clone)]
pub struct AgentCoreHandle {
    pub cmd_tx: mpsc::Sender<AgentCommand>,
    pub event_rx: Arc<Mutex<mpsc::Receiver<AgentEventEnvelope>>>,
    pub busy: Arc<AtomicBool>,
    pub queued: Arc<AtomicUsize>,
}

struct CorePreTurnUi {
    request_id: String,
    peer_id: Option<String>,
    event_tx: mpsc::Sender<AgentEventEnvelope>,
}

impl PreTurnUi for CorePreTurnUi {
    fn compact_notice(&self, path: &str) {
        let _ = self.event_tx.send(AgentEventEnvelope {
            peer_id: self.peer_id.clone(),
            request_id: self.request_id.clone(),
            event: AgentEvent::Warning {
                request_id: self.request_id.clone(),
                message: format!("Compacted. Transcript: {}", path),
            },
        });
    }

    fn background_notification(&self, source: &str, message: &str) {
        let _ = self.event_tx.send(AgentEventEnvelope {
            peer_id: self.peer_id.clone(),
            request_id: self.request_id.clone(),
            event: AgentEvent::Warning {
                request_id: self.request_id.clone(),
                message: format!("[Background {}]: {}", source, message),
            },
        });
    }
}

pub struct AgentCore;

impl AgentCore {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        ctx: AgentContext,
        mut llm: Box<dyn Llm>,
        registry: Arc<HandlerRegistry>,
        tool_defs: Arc<Vec<serde_json::Value>>,
        model_name: String,
        prompts_dir: std::path::PathBuf,
        transcript_dir: std::path::PathBuf,
        projector: Arc<Projector>,
        memory_seed: Arc<MemorySeed>,
        seed_collector: Arc<Mutex<SeedCollector>>,
    ) -> AgentCoreHandle {
        let (cmd_tx, cmd_rx) = mpsc::channel::<AgentCommand>();
        let (event_tx, event_rx) = mpsc::channel::<AgentEventEnvelope>();
        let event_rx = Arc::new(Mutex::new(event_rx));

        // Ensure prompt defaults exist for this data dir.
        {
            let pa = PromptAssembler::new(&prompts_dir);
            pa.init_defaults();
        }

        let busy_flag = Arc::new(AtomicBool::new(false));
        let queued_len = Arc::new(AtomicUsize::new(0));

        // Track current request/peer for EventBus forwarding (sequential core).
        let current_route: Arc<Mutex<Option<(String, Option<String>, bool)>>> =
            Arc::new(Mutex::new(None));

        // Subscribe EventBus once; forward only when a turn is active and includes bus events.
        {
            let route = Arc::clone(&current_route);
            let tx = event_tx.clone();
            ctx.services.event_bus.subscribe(Arc::new(move |record: EventRecord| {
                let (req, peer, include) = match route.lock().ok().and_then(|g| g.clone()) {
                    Some(v) => v,
                    None => return,
                };
                if !include {
                    return;
                }
                let _ = tx.send(AgentEventEnvelope {
                    peer_id: peer.clone(),
                    request_id: req.clone(),
                    event: AgentEvent::BusEvent {
                        request_id: req,
                        event: record.event,
                        data: record.data,
                    },
                });
            }));
        }

        let busy_flag_worker = Arc::clone(&busy_flag);
        let queued_len_worker = Arc::clone(&queued_len);
        std::thread::spawn(move || {
            let mut messages: Vec<serde_json::Value> = ctx.services.sessions.rebuild();
            let mut prev_message_count = messages.len();
            let turn_counter = std::sync::atomic::AtomicUsize::new(0);

            loop {
                let cmd = match cmd_rx.recv() {
                    Ok(c) => c,
                    Err(_) => break,
                };

                match cmd {
                    AgentCommand::Shutdown => break,
                    AgentCommand::Interrupt { .. } => {
                        if let Some(sig) = &ctx.signals.interrupt {
                            sig.store(true, Ordering::Release);
                        }
                    }
                    AgentCommand::Status { request_id, peer_id } => {
                        let _ = event_tx.send(AgentEventEnvelope {
                            peer_id,
                            request_id: request_id.clone(),
                            event: AgentEvent::Status {
                                request_id,
                                busy: busy_flag_worker.load(Ordering::Acquire),
                                queued: queued_len_worker.load(Ordering::Acquire),
                                agent_name: ctx.identity.name.clone(),
                                agent_role: ctx.identity.role.clone(),
                                session_id: ctx.identity.session_id.clone(),
                            },
                        });
                    }
                    AgentCommand::RunTurn {
                        request_id,
                        peer_id,
                        prompt,
                        include_bus_events,
                    } => {
                        queued_len_worker.fetch_update(Ordering::AcqRel, Ordering::Acquire, |x| {
                            Some(x.saturating_sub(1))
                        })
                        .ok();
                        busy_flag_worker.store(true, Ordering::Release);
                        if let Ok(mut r) = current_route.lock() {
                            *r = Some((request_id.clone(), peer_id.clone(), include_bus_events));
                        }

                        let _ = event_tx.send(AgentEventEnvelope {
                            peer_id: peer_id.clone(),
                            request_id: request_id.clone(),
                            event: AgentEvent::TurnStarted {
                                request_id: request_id.clone(),
                            },
                        });

                        // Clear interrupt before each run.
                        if let Some(sig) = &ctx.signals.interrupt {
                            sig.store(false, Ordering::Release);
                        }

                        let turn = turn_counter.fetch_add(1, Ordering::AcqRel) + 1;
                        let turn_key = format!("turn_{}", turn);

                        // Append user message (projected through demand-paging gate).
                        let projected_input = projector.project("user_input", &turn_key, &prompt);
                        messages.push(serde_json::json!({
                            "role": "user",
                            "content": projected_input.as_str(),
                        }));

                        let preturn_ui = CorePreTurnUi {
                            request_id: request_id.clone(),
                            peer_id: peer_id.clone(),
                            event_tx: event_tx.clone(),
                        };

                        let token_count = build_pre_turn_context(
                            &mut messages,
                            &ctx,
                            llm.as_mut(),
                            &transcript_dir,
                            &projector,
                            turn,
                            &memory_seed,
                            &prompt,
                            &preturn_ui,
                        );
                        let _ = token_count;

                        let seed_render = seed_collector
                            .lock()
                            .expect("SeedCollector lock poisoned")
                            .render();
                        let mut prompt_assembler = PromptAssembler::new(&prompts_dir);
                        prompt_assembler.reload();
                        let prompt_ctx = build_prompt_context(
                            &ctx.identity.name,
                            &ctx.identity.role,
                            &ctx.cwd,
                            tool_defs.len(),
                            &model_name,
                            &ctx.identity.id,
                            &ctx.identity.session_id,
                            seed_render,
                        );
                        let system = prompt_assembler.compose(&prompt_ctx);

                        // Tool callback: forward tool events as structured AgentEvents.
                        let tx_for_tools = event_tx.clone();
                        let req_for_tools = request_id.clone();
                        let peer_for_tools = peer_id.clone();
                        let tool_cb: Arc<dyn Fn(ToolEvent) + Send + Sync> =
                            Arc::new(move |evt: ToolEvent| match evt {
                                ToolEvent::Start { name, input } => {
                                    let _ = tx_for_tools.send(AgentEventEnvelope {
                                        peer_id: peer_for_tools.clone(),
                                        request_id: req_for_tools.clone(),
                                        event: AgentEvent::ToolStart {
                                            request_id: req_for_tools.clone(),
                                            name,
                                            input,
                                        },
                                    });
                                }
                                ToolEvent::Complete {
                                    name,
                                    summary,
                                    duration,
                                } => {
                                    let _ = tx_for_tools.send(AgentEventEnvelope {
                                        peer_id: peer_for_tools.clone(),
                                        request_id: req_for_tools.clone(),
                                        event: AgentEvent::ToolComplete {
                                            request_id: req_for_tools.clone(),
                                            name,
                                            summary,
                                            duration_ms: duration.as_millis() as u64,
                                        },
                                    });
                                }
                                ToolEvent::Error {
                                    name,
                                    error,
                                    duration,
                                } => {
                                    let _ = tx_for_tools.send(AgentEventEnvelope {
                                        peer_id: peer_for_tools.clone(),
                                        request_id: req_for_tools.clone(),
                                        event: AgentEvent::ToolError {
                                            request_id: req_for_tools.clone(),
                                            name,
                                            error,
                                            duration_ms: duration.as_millis() as u64,
                                        },
                                    });
                                }
                            });

                        let loop_ctx = AgentContext {
                            tool_callback: Some(tool_cb),
                            ..ctx.clone()
                        };

                        run_agent_loop(
                            llm.as_mut(),
                            &system,
                            &mut messages,
                            &tool_defs,
                            registry.as_ref(),
                            &loop_ctx,
                        );

                        // Persist new messages to session store.
                        if messages.len() > prev_message_count {
                            ctx.services
                                .sessions
                                .append_turn(&messages[prev_message_count..]);
                            prev_message_count = messages.len();
                        }

                        if let Some(text) = extract_last_response_text(&messages) {
                            let _ = event_tx.send(AgentEventEnvelope {
                                peer_id: peer_id.clone(),
                                request_id: request_id.clone(),
                                event: AgentEvent::TurnFinished {
                                    request_id: request_id.clone(),
                                    assistant_text: text,
                                },
                            });
                        } else {
                            let _ = event_tx.send(AgentEventEnvelope {
                                peer_id: peer_id.clone(),
                                request_id: request_id.clone(),
                                event: AgentEvent::Error {
                                    request_id: request_id.clone(),
                                    message: "No assistant response found.".to_string(),
                                },
                            });
                        }

                        if let Ok(mut r) = current_route.lock() {
                            *r = None;
                        }
                        busy_flag_worker.store(false, Ordering::Release);
                    }
                }
            }
        });

        AgentCoreHandle {
            cmd_tx,
            event_rx,
            busy: Arc::clone(&busy_flag),
            queued: Arc::clone(&queued_len),
        }
    }

    /// Convenience for callers to enqueue a turn and increment queue accounting.
    pub fn enqueue_turn(
        handle: &AgentCoreHandle,
        request_id: String,
        peer_id: Option<String>,
        prompt: String,
        include_bus_events: bool,
    ) -> Result<(), String> {
        handle.queued.fetch_add(1, Ordering::AcqRel);
        handle
            .cmd_tx
            .send(AgentCommand::RunTurn {
                request_id,
                peer_id,
                prompt,
                include_bus_events,
            })
            .map_err(|e| format!("AgentCore command send failed: {}", e))
    }
}


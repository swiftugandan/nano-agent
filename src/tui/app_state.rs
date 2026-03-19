use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ratatui::text::Line;

use crate::handler::AgentContext;
use crate::handler::AgentServices;
use crate::types::ToolEvent;

use super::events::LogLevel;
use super::events::UiEvent;
use super::theme::{Theme, ThemeTokens};

#[derive(Clone, Debug)]
pub enum AtRefKind {
    File,
    Dir,
    Missing,
}

#[derive(Clone, Debug)]
pub struct AtRef {
    pub display: String,
    pub resolved: String,
    pub kind: AtRefKind,
}

#[derive(Clone, Debug)]
pub struct AtPickItem {
    pub rel: String,
    pub is_dir: bool,
    pub score: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InspectorTab {
    Status,
    Tasks,
    Team,
    Events,
    Tools,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppFocus {
    Sidebar,
    Chat,
    Inspector,
    Input,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunState {
    Idle,
    Running,
}

#[derive(Clone, Debug)]
pub enum ChatItem {
    User(String),
    Assistant(String),
    ToolStart { name: String, preview: String },
    ToolOk { summary: String, seconds: f64 },
    ToolErr { error: String, seconds: f64 },
    Notice(String),
}

#[derive(Clone, Debug)]
pub struct ActivityItem {
    pub at: Instant,
    pub source: &'static str,
    pub level: LogLevel,
    pub message: String,
}

#[derive(Clone)]
pub struct AppState {
    pub theme: Theme,
    pub no_color: bool,
    pub tokens: ThemeTokens,

    pub services: AgentServices,
    pub agent_name: String,
    pub agent_role: String,
    pub backend: String,
    pub model: String,
    pub session_id: String,

    pub focus: AppFocus,
    pub inspector_tab: InspectorTab,
    pub inspector_visible: bool,
    pub sidebar_visible: bool,
    pub help_open: bool,

    pub run_state: RunState,
    pub started_at: Instant,
    pub last_tick: Instant,
    pub last_status_refresh: Instant,
    pub cached_tokens: usize,
    pub cached_turns: usize,
    pub cached_todo_state: String,
    pub cached_task_list: String,
    pub cached_team_list: String,
    pub cached_events: String,

    pub last_at_refs: Vec<AtRef>,
    pub at_picker_open: bool,
    pub at_anchor: usize,
    pub at_query: String,
    pub at_selection: usize,
    pub at_scroll: usize,
    pub at_results: Vec<AtPickItem>,
    pub input_scroll: u16,

    /// Session-only history for the TUI input buffer (read via Up/Down).
    /// This is intentionally separate from the CLI reedline history.
    pub input_history: Vec<String>,
    /// When browsing history, `Some(i)` points at `input_history[i]`.
    /// When `None`, the user is at the live draft.
    pub input_history_pos: Option<usize>,
    /// Saved live draft so we can restore it when navigating back down.
    pub input_history_draft: Option<(String, usize)>, // (text, cursor_byte_index)

    pub messages: Option<Arc<Mutex<Vec<serde_json::Value>>>>,
    pub chat: Vec<ChatItem>,
    /// Scroll up from bottom, in lines. 0 = anchored to bottom.
    pub chat_scroll: i32,

    pub toasts: Vec<(String, Instant)>,
    pub last_warning: Option<String>,
    pub last_error: Option<String>,

    pub activity: VecDeque<ActivityItem>,
    pub tool_feed: Vec<Line<'static>>,

    pub quit_requested: bool,
}

impl AppState {
    pub fn new(ctx: &AgentContext, model_name: &str, theme: Theme, no_color: bool) -> Self {
        let tokens = theme.tokens(no_color);
        Self {
            theme,
            no_color,
            tokens,
            services: ctx.services.clone(),
            agent_name: ctx.identity.name.clone(),
            agent_role: ctx.identity.role.clone(),
            backend: ctx.llm_backend.clone(),
            model: model_name.to_string(),
            session_id: ctx.identity.session_id.clone(),
            focus: AppFocus::Input,
            inspector_tab: InspectorTab::Status,
            inspector_visible: true,
            sidebar_visible: true,
            help_open: false,
            run_state: RunState::Idle,
            started_at: Instant::now(),
            last_tick: Instant::now(),
            last_status_refresh: Instant::now(),
            cached_tokens: 0,
            cached_turns: 0,
            cached_todo_state: String::new(),
            cached_task_list: String::new(),
            cached_team_list: String::new(),
            cached_events: String::new(),
            last_at_refs: Vec::new(),
            at_picker_open: false,
            at_anchor: 0,
            at_query: String::new(),
            at_selection: 0,
            at_scroll: 0,
            at_results: Vec::new(),
            input_scroll: 0,

            input_history: Vec::new(),
            input_history_pos: None,
            input_history_draft: None,
            messages: None,
            chat: Vec::new(),
            chat_scroll: 0,
            toasts: Vec::new(),
            last_warning: None,
            last_error: None,
            activity: VecDeque::new(),
            tool_feed: Vec::new(),
            quit_requested: false,
        }
    }

    pub fn reset_input_history_nav(&mut self) {
        self.input_history_pos = None;
        self.input_history_draft = None;
    }

    pub fn push_input_history(&mut self, text: &str) {
        const INPUT_HISTORY_MAX: usize = 1000;

        let t = text.trim();
        if t.is_empty() {
            return;
        }

        // Avoid immediate duplicates when re-submitting the same recalled entry.
        if self.input_history.last().is_some_and(|last| last == t) {
            return;
        }

        self.input_history.push(t.to_string());
        if self.input_history.len() > INPUT_HISTORY_MAX {
            let overflow = self.input_history.len() - INPUT_HISTORY_MAX;
            self.input_history.drain(0..overflow);
        }
    }

    pub fn set_messages(&mut self, messages: Arc<Mutex<Vec<serde_json::Value>>>) {
        self.messages = Some(messages);
    }

    pub fn toggle_help(&mut self) {
        self.help_open = !self.help_open;
    }

    pub fn push_toast(&mut self, msg: &str) {
        self.toasts.push((msg.to_string(), Instant::now()));
        if self.toasts.len() > 3 {
            self.toasts.remove(0);
        }
    }

    pub fn push_activity(&mut self, source: &'static str, level: LogLevel, message: String) {
        self.activity.push_back(ActivityItem {
            at: Instant::now(),
            source,
            level,
            message,
        });
        while self.activity.len() > 500 {
            self.activity.pop_front();
        }
    }

    pub fn on_tick(&mut self) {
        // expire toasts after ~2.5s
        let now = Instant::now();
        self.toasts
            .retain(|(_, t)| now.duration_since(*t).as_millis() < 2500);

        // Refresh derived status (token estimate) at a low rate to avoid UI stutter.
        if now.duration_since(self.last_status_refresh).as_millis() >= 500 {
            self.refresh_status();
            self.last_status_refresh = now;
        }
    }

    /// Update cached status using try_lock/try_read so we never block the UI thread
    /// while the agent thread holds locks (e.g. during run_agent_loop).
    pub fn refresh_status(&mut self) {
        if let Some(messages) = self.messages.as_ref() {
            if let Ok(msgs) = messages.try_lock() {
                self.cached_turns = msgs.len();
                self.cached_tokens = crate::memory::estimate_tokens(&msgs);
            }
        }
        if let Ok(todo) = self.services.todo.try_read() {
            self.cached_todo_state = todo.render();
        }
        if let Ok(tasks) = self.services.task_manager.try_read() {
            self.cached_task_list = tasks.list_all();
        }
        if let Ok(team) = self.services.teammate_manager.try_read() {
            self.cached_team_list = team.list_all();
        }
        self.cached_events = self.services.event_bus.list_recent(30);
    }

    pub fn on_resize(&mut self) {
        // renderer decides visibility; just keep scroll in range
        self.chat_scroll = self.chat_scroll.clamp(0, 5000);
    }

    pub fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            AppFocus::Sidebar => AppFocus::Chat,
            AppFocus::Chat => AppFocus::Inspector,
            AppFocus::Inspector => AppFocus::Input,
            AppFocus::Input => AppFocus::Sidebar,
        };
    }

    pub fn scroll_chat(&mut self, delta: i32) {
        self.chat_scroll = (self.chat_scroll + delta).clamp(0, 5000);
    }

    pub fn push_chat_user(&mut self, text: &str) {
        self.chat.push(ChatItem::User(text.to_string()));
    }

    pub fn push_chat_assistant(&mut self, text: &str) {
        self.chat.push(ChatItem::Assistant(text.to_string()));
        self.run_state = RunState::Idle;
    }

    pub fn handle_slash_command(&mut self, cmd: &str) {
        match cmd {
            "/quit" => {
                self.quit_requested = true;
            }
            "/clear" => {
                self.chat.clear();
                self.push_toast("Cleared view (session remains on disk)");
            }
            "/refs" => {
                self.inspector_tab = InspectorTab::Status;
                self.inspector_visible = true;
                if self.last_at_refs.is_empty() {
                    self.push_toast("No @refs captured yet");
                } else {
                    self.push_toast("@refs");
                }
            }
            "/status" => {
                self.inspector_tab = InspectorTab::Status;
                self.inspector_visible = true;
                self.push_toast("Status");
            }
            "/tasks" => {
                self.inspector_tab = InspectorTab::Tasks;
                self.inspector_visible = true;
                self.push_toast("Tasks");
            }
            "/team" => {
                self.inspector_tab = InspectorTab::Team;
                self.inspector_visible = true;
                self.push_toast("Team");
            }
            "/events" => {
                self.inspector_tab = InspectorTab::Events;
                self.inspector_visible = true;
                self.push_toast("Events");
            }
            "/help" => {
                self.help_open = true;
            }
            "/resume" => {
                self.push_toast("Resume is a CLI flag: --resume <id>");
            }
            _ => {
                self.push_toast("Unknown command");
            }
        }
    }

    pub fn apply_event(&mut self, evt: UiEvent) {
        match evt {
            UiEvent::Tool(te) => self.on_tool_event(te),
            UiEvent::Warning(w) => {
                self.last_warning = Some(w.clone());
                self.chat.push(ChatItem::Notice(w));
                self.push_activity(
                    "ui",
                    LogLevel::Warn,
                    self.last_warning.clone().unwrap_or_default(),
                );
            }
            UiEvent::Error(e) => {
                self.last_error = Some(e.clone());
                self.chat.push(ChatItem::Notice(e));
                self.push_activity(
                    "ui",
                    LogLevel::Error,
                    self.last_error.clone().unwrap_or_default(),
                );
                self.run_state = RunState::Idle;
            }
            UiEvent::Toast(t) => self.push_toast(&t),
            UiEvent::Background { source, message } => {
                self.push_toast(&format!("bg {}: {}", source, message));
                self.push_activity(
                    "background",
                    LogLevel::Info,
                    format!("{}: {}", source, message),
                );
            }
            UiEvent::BusEvent { event, .. } => {
                self.push_activity("event", LogLevel::Info, event);
            }
            UiEvent::SubagentProgress { message } => {
                self.push_activity("subagent", LogLevel::Info, message);
            }
            UiEvent::LogLine {
                source,
                level,
                message,
            } => {
                // Source comes from emitter; store as-is but keep a stable label for rendering.
                let src: &'static str = match source.as_str() {
                    "resilience" => "resilience",
                    "context-guard" => "context-guard",
                    _ => "log",
                };
                self.push_activity(src, level, message);
            }
            UiEvent::AgentFinished { assistant_text } => {
                self.push_chat_assistant(&assistant_text);
                self.push_activity("agent", LogLevel::Info, "turn finished".to_string());
            }
        }
    }

    fn on_tool_event(&mut self, evt: ToolEvent) {
        match evt {
            ToolEvent::Start { name, input } => {
                let preview = preview_json(&input);
                self.chat.push(ChatItem::ToolStart { name, preview });
                self.push_activity("tool", LogLevel::Info, "tool started".to_string());
            }
            ToolEvent::Complete {
                name: _,
                summary,
                duration,
            } => {
                self.chat.push(ChatItem::ToolOk {
                    summary,
                    seconds: duration.as_secs_f64(),
                });
                self.push_activity("tool", LogLevel::Info, "tool complete".to_string());
            }
            ToolEvent::Error {
                name: _,
                error,
                duration,
            } => {
                self.chat.push(ChatItem::ToolErr {
                    error,
                    seconds: duration.as_secs_f64(),
                });
                self.push_activity("tool", LogLevel::Error, "tool error".to_string());
            }
        }
    }
}

fn preview_json(v: &serde_json::Value) -> String {
    let s = match v {
        serde_json::Value::Object(obj) => obj
            .iter()
            .take(3)
            .map(|(k, vv)| format!("{}={}", k, short_value(vv)))
            .collect::<Vec<_>>()
            .join(", "),
        other => short_value(other),
    };
    if s.is_empty() {
        "(empty)".to_string()
    } else {
        s
    }
}

fn short_value(v: &serde_json::Value) -> String {
    let s = match v {
        serde_json::Value::String(x) => x.clone(),
        other => other.to_string(),
    };
    let mut out = String::new();
    for ch in s.chars().take(64) {
        out.push(ch);
    }
    if s.chars().count() > 64 {
        out.push('…');
    }
    out
}

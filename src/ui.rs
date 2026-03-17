use std::borrow::Cow;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nu_ansi_term::{Color, Style};
use reedline::{
    Completer, Highlighter, Prompt, PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus,
    Span, StyledText, Suggestion, ValidationResult, Validator,
};

// ---------------------------------------------------------------------------
// ANSI escape codes for raw terminal output (outside reedline)
// ---------------------------------------------------------------------------
const DIM_CYAN: &str = "\x1b[2;36m";
const BOLD_YELLOW: &str = "\x1b[1;33m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const BOLD_RED: &str = "\x1b[1;31m";
const DIM_GRAY: &str = "\x1b[2;37m";
const DIM_MAGENTA: &str = "\x1b[2;35m";
const RESET: &str = "\x1b[0m";

const SPINNER_FRAMES: &[char] = &[
    '\u{280B}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283C}', '\u{2834}', '\u{2826}', '\u{2827}',
    '\u{2807}', '\u{280F}',
];

// ---------------------------------------------------------------------------
// PromptState: shared mutable state for the prompt display
// ---------------------------------------------------------------------------

pub struct PromptState {
    pub agent_name: String,
    pub agent_role: String,
    pub turn: usize,
    pub tokens: usize,
}

impl PromptState {
    pub fn new(agent_name: &str, agent_role: &str) -> Self {
        Self {
            agent_name: agent_name.to_string(),
            agent_role: agent_role.to_string(),
            turn: 0,
            tokens: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// NanoPrompt: implements reedline::Prompt with status bar + indicator
// ---------------------------------------------------------------------------

pub struct NanoPrompt {
    pub state: Arc<Mutex<PromptState>>,
}

impl NanoPrompt {
    pub fn new(state: Arc<Mutex<PromptState>>) -> Self {
        Self { state }
    }
}

impl Prompt for NanoPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        let s = self.state.lock().unwrap();
        let name = Style::new().bold().fg(Color::White).paint(&s.agent_name);
        let role = Style::new()
            .dimmed()
            .fg(Color::Cyan)
            .paint(format!("[{}]", s.agent_role));
        let turn = Style::new()
            .fg(Color::Yellow)
            .paint(format!("t:{}", s.turn));
        let tokens = Style::new()
            .dimmed()
            .fg(Color::Yellow)
            .paint(format!("~{}tk", format_tokens(s.tokens)));
        Cow::Owned(format!("{} {} {} {}", name, role, turn, tokens))
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _mode: PromptEditMode) -> Cow<'_, str> {
        Cow::Owned(format!(
            "\n {}{}{}",
            Style::new().bold().fg(Color::Cyan).prefix(),
            "\u{276F} ",
            Style::new().bold().fg(Color::Cyan).suffix()
        ))
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            " {}: {}",
            Style::new().dimmed().fg(Color::Cyan).prefix(),
            Style::new().dimmed().fg(Color::Cyan).suffix()
        ))
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "(failed) ",
        };
        Cow::Owned(format!("\n (search: {}{}) ", prefix, history_search.term))
    }
}

fn format_tokens(tokens: usize) -> String {
    if tokens >= 1000 {
        format!("{:.1}k", tokens as f64 / 1000.0)
    } else {
        format!("{}", tokens)
    }
}

// ---------------------------------------------------------------------------
// SlashCompleter: tab-completion for slash commands
// ---------------------------------------------------------------------------

pub struct SlashCompleter {
    commands: Vec<String>,
}

impl SlashCompleter {
    pub fn new(commands: Vec<String>) -> Self {
        Self { commands }
    }
}

impl Completer for SlashCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        if !line.starts_with('/') {
            return vec![];
        }
        let prefix = &line[..pos];
        self.commands
            .iter()
            .filter(|c| c.starts_with(prefix))
            .map(|c| Suggestion {
                value: c.clone(),
                description: None,
                style: None,
                extra: None,
                span: Span::new(0, pos),
                append_whitespace: true,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// NanoHighlighter: colors slash commands magenta
// ---------------------------------------------------------------------------

pub struct NanoHighlighter;

impl Highlighter for NanoHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled = StyledText::new();
        if line.starts_with('/') {
            styled.push((Style::new().bold().fg(Color::Magenta), line.to_string()));
        } else {
            styled.push((Style::default(), line.to_string()));
        }
        styled
    }
}

// ---------------------------------------------------------------------------
// NanoValidator: multiline input for unmatched triple-backticks
// ---------------------------------------------------------------------------

pub struct NanoValidator;

impl Validator for NanoValidator {
    fn validate(&self, line: &str) -> ValidationResult {
        let backtick_count = line.matches("```").count();
        if !backtick_count.is_multiple_of(2) {
            ValidationResult::Incomplete
        } else {
            ValidationResult::Complete
        }
    }
}

// ---------------------------------------------------------------------------
// UiRenderer: formatted output for splash, tools, responses, errors
// ---------------------------------------------------------------------------

pub struct StatusInfo<'a> {
    pub tokens: usize,
    pub turns: usize,
    pub session_id: &'a str,
    pub backend: &'a str,
    pub model: &'a str,
    pub uptime: Duration,
    pub todo_state: &'a str,
    pub bg_count: usize,
}

pub struct UiRenderer;

impl UiRenderer {
    pub fn splash_screen(
        backend: &str,
        model: &str,
        agent_name: &str,
        agent_role: &str,
        tool_count: usize,
        session_id: &str,
        history_count: usize,
    ) {
        let border = Style::new().dimmed().fg(Color::Cyan);
        let label = Style::new().dimmed().fg(Color::White);
        let value = Style::new().bold().fg(Color::White);
        let w = 50; // inner box width

        let top = border.paint(format!(" \u{250C}{}\u{2510}", "\u{2500}".repeat(w)));
        let bot = border.paint(format!(" \u{2514}{}\u{2518}", "\u{2500}".repeat(w)));
        let pipe = border.paint("\u{2502}");

        let backend_val = format!("{} ({})", backend, model);
        let agent_val = format!("{} [{}]", agent_name, agent_role);
        let tools_val = format!("{} registered", tool_count);
        let session_short: String = session_id.chars().take(36).collect();
        let history_val = format!("file (~{} entries)", history_count);

        // Pad plain text to target width, then paint — avoids ANSI bytes breaking alignment
        let splash_row = |lbl: &str, val: &str| {
            let lbl_painted = label.paint(format!("{:<10}", lbl));
            let val_painted = value.paint(val);
            let visible_len = lbl.len().max(10) + val.len();
            let pad = (w - 2).saturating_sub(visible_len);
            format!(
                " {}  {}{}{}{}",
                pipe,
                lbl_painted,
                val_painted,
                " ".repeat(pad),
                pipe
            )
        };

        println!("{}", top);
        let title = format!("{:<width$}", "  nano-agent v0.1.0", width = w);
        println!(" {} {}{}", pipe, value.paint(&title), pipe);
        let blank = " ".repeat(w);
        println!(" {} {}{}", pipe, blank, pipe);
        println!("{}", splash_row("backend", &backend_val));
        println!("{}", splash_row("agent", &agent_val));
        println!("{}", splash_row("tools", &tools_val));
        println!("{}", splash_row("session", &session_short));
        println!("{}", splash_row("history", &history_val));
        println!("{}", bot);
        println!();

        let cmds = Style::new().dimmed().fg(Color::White);
        println!(
            "  {}",
            cmds.paint("/quit  /clear  /status  /tasks  /team  /events  /resume  /help")
        );
        println!();
    }

    pub fn show_tool_start(name: &str, input: &serde_json::Value) {
        let tail_len = 40usize.saturating_sub(name.len()).max(2);
        let tail: String = "\u{2500}".repeat(tail_len);
        println!(
            "\n {}\u{2500}\u{2500} tool: {}{}{} {}{}",
            DIM_GRAY, BOLD_YELLOW, name, DIM_GRAY, tail, RESET
        );
        // Show key parameters
        if let Some(obj) = input.as_object() {
            for (k, v) in obj.iter().take(3) {
                let display = match v {
                    serde_json::Value::String(s) => {
                        if s.chars().count() > 60 {
                            let truncated: String = s.chars().take(57).collect();
                            format!("\"{}...\"", truncated)
                        } else {
                            format!("\"{}\"", s)
                        }
                    }
                    other => {
                        let s = other.to_string();
                        if s.chars().count() > 60 {
                            let truncated: String = s.chars().take(57).collect();
                            format!("{}...", truncated)
                        } else {
                            s
                        }
                    }
                };
                println!("   {}{}: {}{}", DIM_GRAY, k, display, RESET);
            }
        }
    }

    pub fn show_tool_complete(name: &str, summary: &str, duration: Duration) {
        println!(
            "   {}\u{2713} {} ({:.1}s){}",
            GREEN,
            summary,
            duration.as_secs_f64(),
            RESET
        );
        let _ = name; // used for context in the header already
    }

    pub fn show_tool_error(name: &str, error: &str, duration: Duration) {
        let short_error = if error.chars().count() > 60 {
            let truncated: String = error.chars().take(57).collect();
            format!("{}...", truncated)
        } else {
            error.to_string()
        };
        println!(
            "   {}\u{2717} {} ({:.1}s){}",
            RED,
            short_error,
            duration.as_secs_f64(),
            RESET
        );
        let _ = name;
    }

    pub fn show_response(text: &str) {
        let mut lines = text.lines();
        if let Some(first) = lines.next() {
            println!("\n {}\u{25C6}{} {}", DIM_CYAN, RESET, first);
            for line in lines {
                println!("   {}", line);
            }
        }
        println!();
        // Thin separator
        println!(" {}{}{}", DIM_GRAY, "\u{2500}".repeat(49), RESET);
    }

    pub fn show_error(msg: &str) {
        println!(" {}\u{2717} {}{}", BOLD_RED, msg, RESET);
    }

    pub fn show_warning(msg: &str) {
        println!(" {}\u{26A0} {}{}", BOLD_YELLOW, msg, RESET);
    }

    pub fn show_status(info: &StatusInfo) {
        let StatusInfo {
            tokens,
            turns,
            session_id,
            ref backend,
            ref model,
            uptime,
            todo_state,
            bg_count,
        } = *info;
        let border = Style::new().dimmed().fg(Color::Cyan);
        let value = Style::new().bold().fg(Color::White);
        let w = 45; // inner box width

        let top = border.paint(format!(
            " \u{250C} Status {}\u{2510}",
            "\u{2500}".repeat(w - 8)
        ));
        let bot = border.paint(format!(" \u{2514}{}\u{2518}", "\u{2500}".repeat(w)));
        let pipe = border.paint("\u{2502}");

        let mins = uptime.as_secs() / 60;
        let secs = uptime.as_secs() % 60;
        let uptime_str = format!("{}m {}s", mins, secs);

        // Pad plain text to target width, then paint
        let status_row = |lbl: &str, val: &str| {
            let content = format!(" {:<12}{}", lbl, val);
            let padded = format!("{:<width$}", content, width = w);
            format!(" {} {}{}", pipe, value.paint(&padded), pipe)
        };

        println!("{}", top);
        println!(
            "{}",
            status_row("tokens", &format!("~{}", format_tokens(tokens)))
        );
        println!("{}", status_row("turns", &format!("{}", turns)));
        println!(
            "{}",
            status_row("session", &session_id.chars().take(34).collect::<String>())
        );
        println!(
            "{}",
            status_row("backend", &format!("{} ({})", backend, model))
        );
        println!("{}", status_row("uptime", &uptime_str));
        if !todo_state.is_empty() && todo_state != "(empty)" {
            let blank = format!("{:<width$}", "", width = w);
            println!(" {} {}{}", pipe, blank, pipe);
            let todo_hdr = format!(" {:<width$}", "Todo:", width = w - 1);
            println!(" {} {}{}", pipe, value.paint(&todo_hdr), pipe);
            for line in todo_state.lines().take(10) {
                let truncated = if line.chars().count() > w - 3 {
                    let t: String = line.chars().take(w - 6).collect();
                    format!("{}...", t)
                } else {
                    line.to_string()
                };
                let padded = format!("   {:<width$}", truncated, width = w - 3);
                println!(" {} {}{}", pipe, padded, pipe);
            }
        }
        if bg_count > 0 {
            println!(
                "{}",
                status_row("background", &format!("{} completed", bg_count))
            );
        }
        println!("{}", bot);
        println!();
    }

    pub fn show_background_notification(source: &str, message: &str) {
        println!(
            "\n {}\u{2591} [bg] {} completed: \"{}\"{}",
            DIM_MAGENTA, source, message, RESET
        );
    }

    pub fn show_compact_notice(path: &str) {
        println!(
            " {}\u{26A0} Context overflow. Compacting...{}",
            BOLD_YELLOW, RESET
        );
        println!(
            " {}\u{2713} Compacted. Transcript saved: {}{}",
            GREEN, path, RESET
        );
    }
}

// ---------------------------------------------------------------------------
// SpinnerHandle: animated braille spinner on a background thread
// ---------------------------------------------------------------------------

pub struct SpinnerHandle {
    active: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    label: Arc<Mutex<String>>,
    output_lock: Arc<Mutex<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SpinnerHandle {
    pub fn start(initial_label: &str) -> Self {
        let active = Arc::new(AtomicBool::new(true));
        let paused = Arc::new(AtomicBool::new(false));
        let label = Arc::new(Mutex::new(initial_label.to_string()));
        let output_lock = Arc::new(Mutex::new(()));

        let a = Arc::clone(&active);
        let p = Arc::clone(&paused);
        let l = Arc::clone(&label);
        let o = Arc::clone(&output_lock);

        let thread = std::thread::spawn(move || {
            let mut frame = 0;
            while a.load(Ordering::SeqCst) {
                if !p.load(Ordering::SeqCst) {
                    let _lock = o.lock().unwrap();
                    // Double-check after acquiring lock
                    if !p.load(Ordering::SeqCst) && a.load(Ordering::SeqCst) {
                        let lbl = l.lock().unwrap().clone();
                        print!(
                            "\r {}{}{} {}...   ",
                            DIM_CYAN, SPINNER_FRAMES[frame], RESET, lbl
                        );
                        std::io::stdout().flush().ok();
                        frame = (frame + 1) % SPINNER_FRAMES.len();
                    }
                }
                std::thread::sleep(Duration::from_millis(80));
            }
            // Clear the spinner line
            print!("\r{}\r", " ".repeat(60));
            std::io::stdout().flush().ok();
        });

        Self {
            active,
            paused,
            label,
            output_lock,
            thread: Some(thread),
        }
    }

    pub fn update_label(&self, new_label: &str) {
        *self.label.lock().unwrap() = new_label.to_string();
    }

    /// Pause spinner and clear its line. Safe to print tool output after this.
    pub fn pause_and_clear(&self) {
        self.paused.store(true, Ordering::SeqCst);
        let _lock = self.output_lock.lock().unwrap();
        print!("\r{}\r", " ".repeat(60));
        std::io::stdout().flush().ok();
    }

    /// Resume spinner animation.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    /// Stop the spinner and join the background thread.
    pub fn stop(&self) {
        self.active.store(false, Ordering::SeqCst);
    }
}

impl Drop for SpinnerHandle {
    fn drop(&mut self) {
        self.active.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

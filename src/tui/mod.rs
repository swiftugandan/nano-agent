pub mod app_state;
pub mod events;
pub mod input;
pub mod render;
pub mod theme;

use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use std::path::{Path, PathBuf};

use crate::handler::AgentContext;
use crate::types::ToolEvent;

use app_state::{AppFocus, AppState, RunState};
use events::{UiEvent, UiEventSender};
use input::{CommandPalette, InputBuffer};
use render::render_frame;
use theme::Theme;

fn parse_at_refs(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '@' {
            continue;
        }

        // @"path with spaces"
        if matches!(chars.peek(), Some('"')) {
            let _ = chars.next(); // consume opening quote
            let mut p = String::new();
            for c in chars.by_ref() {
                if c == '"' {
                    break;
                }
                p.push(c);
            }
            let p = p.trim().to_string();
            if !p.is_empty() {
                out.push(p);
            }
            continue;
        }

        // @path (read until whitespace)
        let mut p = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_whitespace() {
                break;
            }
            p.push(c);
            chars.next();
        }
        let p = p.trim().trim_end_matches(|c: char| {
            matches!(
                c,
                ',' | '.' | ';' | ':' | ')' | ']' | '}' | '>' | '"' | '\''
            )
        });
        if !p.is_empty() {
            out.push(p.to_string());
        }
    }

    // Dedup while preserving order
    let mut uniq = Vec::new();
    for p in out {
        if !uniq.contains(&p) {
            uniq.push(p);
        }
    }
    uniq
}

fn fuzzy_score(hay: &str, needle: &str) -> Option<i32> {
    let h = hay.to_lowercase();
    let n = needle.to_lowercase();
    if n.is_empty() {
        return Some(0);
    }
    if let Some(idx) = h.find(&n) {
        // Prefer earlier matches; cap to keep scores in a sane range.
        return Some(10_000 - (idx as i32).min(9_000));
    }
    // Subsequence match: characters in order.
    let mut it = h.chars();
    let mut matched = 0i32;
    for c in n.chars() {
        let mut found = false;
        for hc in it.by_ref() {
            if hc == c {
                found = true;
                matched += 1;
                break;
            }
        }
        if !found {
            return None;
        }
    }
    Some(100 + matched)
}

fn build_at_results(cwd: &Path, query: &str) -> Vec<app_state::AtPickItem> {
    let mut out: Vec<app_state::AtPickItem> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![cwd.to_path_buf()];
    let mut visited = 0usize;
    let mut produced = 0usize;
    let max_visited = 8_000usize;
    let max_results = 200usize;

    while let Some(dir) = stack.pop() {
        if visited >= max_visited || produced >= max_results {
            break;
        }
        let rd = match std::fs::read_dir(&dir) {
            Ok(x) => x,
            Err(_) => continue,
        };
        for entry in rd {
            if visited >= max_visited || produced >= max_results {
                break;
            }
            visited += 1;
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            let path = entry.path();
            let md = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let is_dir = md.is_dir();
            if is_dir {
                stack.push(path.clone());
            }
            let rel = match path.strip_prefix(cwd) {
                Ok(p) => p.to_string_lossy().to_string(),
                Err(_) => continue,
            };
            let score = match fuzzy_score(&rel, query) {
                Some(s) => s,
                None => continue,
            };
            out.push(app_state::AtPickItem { rel, is_dir, score });
            produced += 1;
        }
    }

    out.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.rel.cmp(&b.rel)));
    out.truncate(max_results);
    out
}

fn at_token(rel: &str, is_dir: bool) -> String {
    let mut p = rel.to_string();
    if is_dir && !p.ends_with('/') {
        p.push('/');
    }
    if p.contains(' ') {
        format!("@\"{}\"", p)
    } else {
        format!("@{}", p)
    }
}

fn slice_by_bytes(s: &str, start: usize, end: usize) -> String {
    // Cursor positions are byte indices; only slice at char boundaries.
    let len = s.len();
    let mut s_idx = start.min(len);
    let mut e_idx = end.min(len);
    if s_idx > e_idx {
        std::mem::swap(&mut s_idx, &mut e_idx);
    }
    while s_idx > 0 && !s.is_char_boundary(s_idx) {
        s_idx -= 1;
    }
    while e_idx < len && !s.is_char_boundary(e_idx) {
        e_idx += 1;
    }
    s[s_idx..e_idx].to_string()
}

pub struct TuiOptions {
    pub no_color: bool,
    pub theme: Theme,
}

pub struct TuiRuntime {
    pub ui_tx: UiEventSender,
}

impl TuiRuntime {
    pub fn new(ui_tx: UiEventSender) -> Self {
        Self { ui_tx }
    }

    /// Tool callback for the agent loop: forward to TUI.
    pub fn tool_callback(&self) -> Arc<dyn Fn(ToolEvent) + Send + Sync> {
        let tx = self.ui_tx.clone();
        Arc::new(move |evt: ToolEvent| {
            let _ = tx.send(UiEvent::Tool(evt));
        })
    }
}

pub fn run_tui(
    ctx: AgentContext,
    messages: Arc<Mutex<Vec<serde_json::Value>>>,
    model_name: String,
    opts: TuiOptions,
    ui_tx: UiEventSender,
    ui_rx: mpsc::Receiver<UiEvent>,
    // Agent thread launcher injected by main; allows reusing existing wiring
    mut launch_agent_turn: impl FnMut(String, Arc<dyn Fn(ToolEvent) + Send + Sync>) + Send + 'static,
) -> std::io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    execute!(stdout, EnableBracketedPaste)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let runtime = TuiRuntime::new(ui_tx.clone());

    let mut app = AppState::new(&ctx, &model_name, opts.theme, opts.no_color);
    app.set_messages(messages);
    app.refresh_status();

    // Basic help hint: keep minimal in-state; rendered by renderer
    app.last_tick = Instant::now();

    let tick_rate = Duration::from_millis(33);
    let mut input = InputBuffer::new();
    let mut palette = CommandPalette::new();

    'outer: loop {
        // Drain UI events
        while let Ok(evt) = ui_rx.try_recv() {
            app.apply_event(evt);
        }

        // Only show cursor when input is focused and no overlay is open.
        let overlays_open = app.help_open || palette.is_open() || app.at_picker_open;
        if app.focus == AppFocus::Input && !overlays_open {
            terminal.show_cursor()?;
        } else {
            terminal.hide_cursor()?;
        }

        terminal.draw(|f| render_frame(f, &mut app, &input, &palette))?;

        // Tick-based animations / timeouts
        if app.last_tick.elapsed() >= tick_rate {
            app.last_tick = Instant::now();
            app.on_tick();
        }

        // Input handling (non-blocking)
        if event::poll(Duration::from_millis(10))? {
            match event::read()? {
                Event::Resize(_, _) => {
                    app.on_resize();
                }
                Event::Paste(data) => {
                    if app.focus == AppFocus::Input {
                        app.reset_input_history_nav();
                        input.insert_str(&data);
                        if app.at_picker_open {
                            if input.cursor() <= app.at_anchor {
                                app.at_picker_open = false;
                            } else {
                                app.at_query = slice_by_bytes(
                                    input.text(),
                                    app.at_anchor.saturating_add(1),
                                    input.cursor(),
                                );
                                app.at_results = build_at_results(&ctx.cwd, &app.at_query);
                                app.at_scroll = 0;
                                app.at_selection = 0;
                            }
                        }
                    }
                }
                Event::Key(key) => {
                    if handle_key(
                        &ctx,
                        key,
                        &mut app,
                        &mut input,
                        &mut palette,
                        &runtime,
                        &mut launch_agent_turn,
                    )? {
                        break 'outer;
                    }
                    if app.quit_requested {
                        break 'outer;
                    }
                }
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), DisableBracketedPaste)?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn handle_key(
    ctx: &AgentContext,
    key: KeyEvent,
    app: &mut AppState,
    input: &mut InputBuffer,
    palette: &mut CommandPalette,
    runtime: &TuiRuntime,
    launch_agent_turn: &mut impl FnMut(String, Arc<dyn Fn(ToolEvent) + Send + Sync>),
) -> std::io::Result<bool> {
    // Global quit
    if key.code == KeyCode::Char('q') && key.modifiers.contains(KeyModifiers::CONTROL) {
        return Ok(true);
    }

    // Global help
    if key.code == KeyCode::Char('?') || key.code == KeyCode::F(1) {
        app.toggle_help();
        return Ok(false);
    }

    // Focus cycle
    if key.code == KeyCode::Tab && key.modifiers.is_empty() {
        app.cycle_focus();
        return Ok(false);
    }

    // Interrupt/cancel (Ctrl+C): if agent running, request interrupt; otherwise clear input.
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        if app.run_state == RunState::Running {
            if let Some(sig) = &ctx.signals.interrupt {
                sig.store(true, Ordering::Release);
            }
            app.push_toast("Interrupt requested");
        } else {
            input.clear();
            app.reset_input_history_nav();
        }
        return Ok(false);
    }

    // Esc closes overlays
    if key.code == KeyCode::Esc {
        palette.close();
        app.help_open = false;
        app.at_picker_open = false;
        return Ok(false);
    }

    // : = panels/toggles (always); / = session/UI commands (when input empty; also typeable as /cmd in input)
    if matches!(
        key,
        KeyEvent {
            code: KeyCode::Char(':'),
            modifiers: KeyModifiers::NONE,
            ..
        }
    ) {
        palette.open_colon();
        return Ok(false);
    }
    if matches!(
        key,
        KeyEvent {
            code: KeyCode::Char('/'),
            modifiers: KeyModifiers::NONE,
            ..
        }
    ) && input.is_empty()
    {
        palette.open_slash();
        return Ok(false);
    }

    // Palette interaction (minimal v1)
    if palette.is_open() && palette.handle_key(key, app) {
        return Ok(false);
    }

    // If focus is not input, don't route typing/editor keys into input.
    // (Chat/Inspector/Sidebar can add their own key handling later.)
    if app.focus != AppFocus::Input {
        match key {
            KeyEvent {
                code: KeyCode::Char(_),
                ..
            }
            | KeyEvent {
                code: KeyCode::Backspace,
                modifiers: KeyModifiers::NONE,
                ..
            }
            | KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::NONE,
                ..
            }
            | KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::NONE,
                ..
            }
            | KeyEvent {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::NONE,
                ..
            }
            | KeyEvent {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::SHIFT,
                ..
            } => return Ok(false),
            _ => {}
        }
    }

    // Input/editor interaction
    match key {
        KeyEvent {
            code: KeyCode::Up,
            modifiers: KeyModifiers::NONE,
            ..
        } if app.at_picker_open && app.focus == AppFocus::Input => {
            if !app.at_results.is_empty() {
                app.at_scroll = app.at_scroll.saturating_sub(1);
                app.at_selection = app.at_scroll;
            }
        }
        KeyEvent {
            code: KeyCode::Up,
            modifiers: KeyModifiers::NONE,
            ..
        } if !app.at_picker_open && app.focus == AppFocus::Input => {
            if app.input_history.is_empty() {
                return Ok(false);
            }

            let last_idx = app.input_history.len() - 1;
            app.input_history_pos = match app.input_history_pos {
                None => Some(last_idx),
                Some(0) => Some(0),
                Some(i) => Some(i.saturating_sub(1)),
            };

            if let Some(pos) = app.input_history_pos {
                // Save live draft once, the first time the user enters history browsing.
                if app.input_history_draft.is_none() {
                    app.input_history_draft = Some((input.text().to_string(), input.cursor()));
                }

                // Clone to avoid holding a borrow into `app.input_history` while mutating `app`.
                let text = app.input_history[pos].clone();
                let cursor = text.len();

                input.clear();
                input.insert_str(&text);
                input.set_cursor(cursor);
                app.input_scroll = 0;
            }
        }
        KeyEvent {
            code: KeyCode::Down,
            modifiers: KeyModifiers::NONE,
            ..
        } if app.at_picker_open && app.focus == AppFocus::Input => {
            if !app.at_results.is_empty() {
                let last = app.at_results.len().saturating_sub(1);
                app.at_scroll = (app.at_scroll + 1).min(last);
                app.at_selection = app.at_scroll;
            }
        }
        KeyEvent {
            code: KeyCode::Down,
            modifiers: KeyModifiers::NONE,
            ..
        } if !app.at_picker_open && app.focus == AppFocus::Input => {
            if app.input_history.is_empty() {
                return Ok(false);
            }

            let len = app.input_history.len();
            match app.input_history_pos {
                None => {
                    // If we haven't started browsing yet, Down restores the live draft.
                    if let Some((draft, cursor)) = app.input_history_draft.take() {
                        input.clear();
                        input.insert_str(&draft);
                        input.set_cursor(cursor);
                        app.input_history_pos = None;
                        app.input_scroll = 0;
                    }
                }
                Some(pos) => {
                    if pos + 1 < len {
                        app.input_history_pos = Some(pos + 1);
                        // Clone to avoid holding a borrow into `app.input_history` while mutating `app`.
                        let text = app.input_history[pos + 1].clone();
                        let cursor = text.len();

                        input.clear();
                        input.insert_str(&text);
                        input.set_cursor(cursor);
                        app.input_scroll = 0;
                    } else {
                        // Past newest: restore live draft and exit history browsing.
                        if let Some((draft, cursor)) = app.input_history_draft.take() {
                            input.clear();
                            input.insert_str(&draft);
                            input.set_cursor(cursor);
                        } else {
                            input.clear();
                        }
                        app.input_history_pos = None;
                        app.input_scroll = 0;
                    }
                }
            }
        }
        KeyEvent {
            code: KeyCode::Enter,
            modifiers: KeyModifiers::SHIFT,
            ..
        } => {
            app.reset_input_history_nav();
            input.newline();
        }
        KeyEvent {
            code: KeyCode::Enter,
            modifiers: KeyModifiers::NONE,
            ..
        } => {
            if app.at_picker_open && app.focus == AppFocus::Input {
                if let Some(it) = app.at_results.get(app.at_scroll) {
                    let token = at_token(&it.rel, it.is_dir);
                    app.reset_input_history_nav();
                    input.replace_range(app.at_anchor, input.cursor(), &token);
                }
                app.at_picker_open = false;
                return Ok(false);
            }

            let current = input.text().to_string();
            if current.trim().is_empty() {
                return Ok(false);
            }

            // Slash compat: route as command (changes panes/state) even while running.
            if current.trim_start().starts_with('/') {
                app.push_input_history(&current);
                app.reset_input_history_nav();
                let text = input.take();
                app.handle_slash_command(text.trim());
                return Ok(false);
            }

            // While running, keep draft text in the input but disable sending.
            if app.run_state == RunState::Running {
                app.push_toast("Turn running — send disabled");
                return Ok(false);
            }

            app.push_input_history(&current);
            app.reset_input_history_nav();

            // Start an agent turn
            let text = input.take();

            // Capture @refs (reference-only). Validate existence without reading contents.
            let refs = parse_at_refs(&text);
            let mut at_refs = Vec::new();
            for r in refs {
                let resolved = if r.starts_with('/') {
                    PathBuf::from(&r)
                } else {
                    ctx.cwd.join(&r)
                };
                let (kind, missing) = match std::fs::metadata(&resolved) {
                    Ok(md) if md.is_dir() => (app_state::AtRefKind::Dir, false),
                    Ok(_md) => (app_state::AtRefKind::File, false),
                    Err(_) => (app_state::AtRefKind::Missing, true),
                };
                if missing {
                    app.push_toast(&format!("Missing @file: {}", r));
                }
                at_refs.push(app_state::AtRef {
                    display: r,
                    resolved: resolved.to_string_lossy().to_string(),
                    kind,
                });
            }
            app.last_at_refs = at_refs;

            app.run_state = RunState::Running;
            app.push_chat_user(&text);
            launch_agent_turn(text, runtime.tool_callback());
        }
        KeyEvent {
            code: KeyCode::Backspace,
            modifiers: KeyModifiers::NONE,
            ..
        } => {
            input.backspace();
            app.reset_input_history_nav();
            if app.at_picker_open && app.focus == AppFocus::Input {
                if input.cursor() <= app.at_anchor {
                    app.at_picker_open = false;
                } else {
                    app.at_query = slice_by_bytes(
                        input.text(),
                        app.at_anchor.saturating_add(1),
                        input.cursor(),
                    );
                    app.at_results = build_at_results(&ctx.cwd, &app.at_query);
                    app.at_scroll = 0;
                    app.at_selection = 0;
                }
            }
        }
        KeyEvent {
            code: KeyCode::Left,
            modifiers: KeyModifiers::NONE,
            ..
        } => input.move_left(),
        KeyEvent {
            code: KeyCode::Right,
            modifiers: KeyModifiers::NONE,
            ..
        } => input.move_right(),
        KeyEvent {
            code: KeyCode::Up,
            modifiers: KeyModifiers::NONE,
            ..
        } => {
            if app.focus == AppFocus::Chat {
                app.scroll_chat(3);
            }
        }
        KeyEvent {
            code: KeyCode::Down,
            modifiers: KeyModifiers::NONE,
            ..
        } => {
            if app.focus == AppFocus::Chat {
                app.scroll_chat(-3);
            }
        }
        KeyEvent {
            code: KeyCode::PageUp,
            modifiers: KeyModifiers::NONE,
            ..
        } => app.scroll_chat(12),
        KeyEvent {
            code: KeyCode::PageDown,
            modifiers: KeyModifiers::NONE,
            ..
        } => app.scroll_chat(-12),
        KeyEvent {
            code: KeyCode::Char(ch),
            modifiers,
            ..
        } if !modifiers.contains(KeyModifiers::CONTROL)
            && !modifiers.contains(KeyModifiers::ALT) =>
        {
            app.reset_input_history_nav();
            input.insert_char(ch);
            if app.focus == AppFocus::Input {
                if ch == '@' {
                    app.at_picker_open = true;
                    app.at_anchor = input.cursor().saturating_sub(1);
                    app.at_query.clear();
                    app.at_scroll = 0;
                    app.at_selection = 0;
                    app.at_results = build_at_results(&ctx.cwd, "");
                } else if app.at_picker_open {
                    if input.cursor() <= app.at_anchor {
                        app.at_picker_open = false;
                    } else {
                        app.at_query = slice_by_bytes(
                            input.text(),
                            app.at_anchor.saturating_add(1),
                            input.cursor(),
                        );
                        app.at_results = build_at_results(&ctx.cwd, &app.at_query);
                        app.at_scroll = 0;
                        app.at_selection = 0;
                    }
                }
            }
        }
        _ => {}
    }

    Ok(false)
}

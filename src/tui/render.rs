use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::*;
use ratatui::widgets::*;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::app_state::{AppFocus, AppState, ChatItem, InspectorTab, RunState};
use super::input::{CommandPalette, InputBuffer, PaletteMode};

pub fn render_frame(
    f: &mut Frame,
    app: &mut AppState,
    input: &InputBuffer,
    palette: &CommandPalette,
) {
    let size = f.area();
    // Background fill
    f.render_widget(
        Block::default().style(Style::default().bg(app.tokens.bg)),
        size,
    );

    let (sidebar_w, inspector_w) = responsive_columns(size);
    let mut constraints = vec![];
    if app.sidebar_visible && sidebar_w > 0 {
        constraints.push(Constraint::Length(sidebar_w));
    }
    constraints.push(Constraint::Min(40));
    if app.inspector_visible && inspector_w > 0 {
        constraints.push(Constraint::Length(inspector_w));
    }

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(size);

    // Column indices depend on collapsed panes
    let mut idx = 0usize;
    let sidebar_area = if app.sidebar_visible && sidebar_w > 0 {
        let a = cols[idx];
        idx += 1;
        Some(a)
    } else {
        None
    };
    let center_area = cols[idx];
    idx += 1;
    let inspector_area = if app.inspector_visible && inspector_w > 0 {
        Some(cols[idx])
    } else {
        None
    };

    if let Some(area) = sidebar_area {
        render_sidebar(f, &*app, area);
    }
    render_center(f, app, input, palette, center_area);
    if let Some(area) = inspector_area {
        render_inspector(f, &*app, area);
    }

    render_toasts(f, &*app, size);
    if app.help_open {
        render_help(f, &*app, size);
    }
    if palette.is_open() {
        render_palette(f, &*app, palette, size);
    }
}

fn responsive_columns(size: Rect) -> (u16, u16) {
    if size.width < 80 {
        return (0, 0);
    }
    if size.width < 100 {
        return (22, 0);
    }
    (26, 34)
}

fn panel<'a>(app: &AppState, title: &'a str, focused: bool) -> Block<'a> {
    let mut base = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(app.tokens.title)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .style(Style::default().bg(app.tokens.panel_bg).fg(app.tokens.text));
    let border_style = if focused {
        Style::default()
            .fg(app.tokens.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(app.tokens.border)
    };
    base = base.border_style(border_style);
    base
}

fn render_sidebar(f: &mut Frame, app: &AppState, area: Rect) {
    let focused = app.focus == AppFocus::Sidebar;
    let lines = vec![
        Line::from(vec![
            Span::styled(
                format!("{} ", app.agent_name),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("[{}]", app.agent_role)),
        ]),
        Line::from(Span::raw("")),
        Line::from(Span::raw(format!("backend: {}", app.backend))),
        Line::from(Span::raw(format!("model:   {}", app.model))),
        Line::from(Span::raw(format!(
            "session: {}",
            app.session_id.chars().take(12).collect::<String>()
        ))),
        Line::from(Span::raw("")),
        Line::from(Span::raw(match app.run_state {
            RunState::Idle => "state: idle",
            RunState::Running => "state: running",
        })),
        Line::from(Span::raw("")),
        Line::from(Span::raw("keys:")),
        Line::from(Span::raw("  Ctrl+Q quit")),
        Line::from(Span::raw("  Ctrl+C interrupt")),
        Line::from(Span::raw("  ? help")),
        Line::from(Span::raw("  Tab focus")),
        Line::from(Span::raw("  /  UI commands (help, clear, quit)")),
        Line::from(Span::raw("  :  panels (tools, inspector, sidebar)")),
    ];
    let w = Paragraph::new(lines).block(panel(app, " nano-agent ", focused));
    f.render_widget(w, area);
}

fn render_center(
    f: &mut Frame,
    app: &mut AppState,
    input: &InputBuffer,
    palette: &CommandPalette,
    area: Rect,
) {
    let input_h = if app.at_picker_open { 9 } else { 5 };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(input_h)])
        .split(area);

    render_chat(f, &*app, chunks[0]);
    render_input(f, app, input, palette, chunks[1]);
}

fn render_chat(f: &mut Frame, app: &AppState, area: Rect) {
    let focused = app.focus == AppFocus::Chat;
    let mut lines: Vec<Line> = Vec::new();
    for item in &app.chat {
        match item {
            ChatItem::User(t) => {
                lines.extend(format_message_block("You", t, area.width as usize));
            }
            ChatItem::Assistant(t) => {
                lines.extend(format_message_block("Agent", t, area.width as usize));
            }
            ChatItem::ToolStart { name, preview } => {
                lines.extend(format_tool_card_start(name, preview, area.width as usize));
            }
            ChatItem::ToolOk { summary, seconds } => {
                lines.extend(format_tool_card_done(
                    true,
                    summary,
                    *seconds,
                    area.width as usize,
                    app.tokens.ok,
                    app.tokens.err,
                ));
            }
            ChatItem::ToolErr { error, seconds } => {
                lines.extend(format_tool_card_done(
                    false,
                    error,
                    *seconds,
                    area.width as usize,
                    app.tokens.ok,
                    app.tokens.err,
                ));
            }
            ChatItem::Notice(t) => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "!",
                        Style::default()
                            .fg(app.tokens.warn)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::raw(truncate_line(t, area.width.saturating_sub(4) as usize)),
                ]));
            }
        }
        lines.push(Line::from(Span::raw("")));
    }

    // Chat scroll is expressed as "lines up from bottom"
    let visible = area.height.saturating_sub(2) as usize; // account for borders
    let total = lines.len();
    let scroll_up = app.chat_scroll.max(0) as usize;
    let bottom_offset = total.saturating_sub(visible);
    let offset = bottom_offset.saturating_sub(scroll_up);

    let para = Paragraph::new(lines)
        .block(panel(app, " chat ", focused))
        .wrap(Wrap { trim: false })
        .scroll((offset.min(u16::MAX as usize) as u16, 0));
    f.render_widget(para, area);
}

fn cursor_metrics(text: &str, cursor: usize, width: u16) -> (u16, u16) {
    let w = width.max(1) as usize;
    let mut row: u16 = 0;
    let mut col: usize = 0;

    let mut idx = 0usize;
    for ch in text.chars() {
        let next = idx + ch.len_utf8();
        let before_cursor = next <= cursor;
        idx = next;

        if ch == '\n' {
            if before_cursor {
                row = row.saturating_add(1);
                col = 0;
            }
            continue;
        }

        let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        if col + cw > w && before_cursor {
            row = row.saturating_add(1);
            col = 0;
        }
        if before_cursor {
            col = (col + cw).min(w);
        }
    }

    if col >= w {
        (row.saturating_add(1), 0)
    } else {
        let cursor_col = (col.min(w.saturating_sub(1))) as u16;
        (row, cursor_col)
    }
}

/// Hard-wrap input text by display width; preserves explicit `\n`. Matches cursor_metrics so
/// rendered lines align with cursor row/col.
fn input_hard_wrap(text: &str, width: u16) -> Vec<Line<'static>> {
    let w = width.max(1) as usize;
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut col = 0usize;
    for ch in text.chars() {
        if ch == '\n' {
            out.push(std::mem::take(&mut cur));
            col = 0;
            continue;
        }
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        if col + cw > w && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            col = 0;
        }
        cur.push(ch);
        col += cw;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out.into_iter().map(|s| Line::from(Span::raw(s))).collect()
}

fn render_input(
    f: &mut Frame,
    app: &mut AppState,
    input: &InputBuffer,
    palette: &CommandPalette,
    area: Rect,
) {
    let focused = app.focus == AppFocus::Input;
    let title = match app.run_state {
        RunState::Idle => " input ",
        RunState::Running => " input (running) ",
    };
    if !app.at_picker_open || area.height < 7 {
        let b = panel(app, title, focused);
        let inner = b.inner(area);
        let (cursor_row, cursor_col) = cursor_metrics(input.text(), input.cursor(), inner.width);
        let visible_h = inner.height.max(1);
        let max_scroll = cursor_row.saturating_sub(visible_h.saturating_sub(1));
        if cursor_row < app.input_scroll {
            app.input_scroll = cursor_row;
        } else if cursor_row >= app.input_scroll + visible_h.saturating_sub(1) {
            app.input_scroll = max_scroll;
        }

        let lines = input_hard_wrap(input.text(), inner.width);
        let p = Paragraph::new(lines).block(b).scroll((app.input_scroll, 0));
        f.render_widget(p, area);

        let overlays_open = app.help_open || palette.is_open() || app.at_picker_open;
        if focused && !overlays_open && inner.width > 0 && inner.height > 0 {
            let y = inner.y + cursor_row.saturating_sub(app.input_scroll);
            let x = inner.x + cursor_col;
            if y < inner.y + inner.height {
                f.set_cursor_position((x, y));
            }
        }
        return;
    }

    // Picker on top, input below.
    let picker_h = area.height.min(10).saturating_sub(4).max(4);
    let input_h = area.height.saturating_sub(picker_h);
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(picker_h), Constraint::Length(input_h)])
        .split(area);

    render_at_picker(f, &*app, parts[0]);

    let b = panel(app, title, focused);
    let inner = b.inner(parts[1]);
    let (cursor_row, cursor_col) = cursor_metrics(input.text(), input.cursor(), inner.width);
    let visible_h = inner.height.max(1);
    let max_scroll = cursor_row.saturating_sub(visible_h.saturating_sub(1));
    if cursor_row < app.input_scroll {
        app.input_scroll = cursor_row;
    } else if cursor_row >= app.input_scroll + visible_h.saturating_sub(1) {
        app.input_scroll = max_scroll;
    }

    let lines = input_hard_wrap(input.text(), inner.width);
    let p = Paragraph::new(lines).block(b).scroll((app.input_scroll, 0));
    f.render_widget(p, parts[1]);

    let overlays_open = app.help_open || palette.is_open() || app.at_picker_open;
    if focused && !overlays_open && inner.width > 0 && inner.height > 0 {
        let y = inner.y + cursor_row.saturating_sub(app.input_scroll);
        let x = inner.x + cursor_col;
        if y < inner.y + inner.height {
            f.set_cursor_position((x, y));
        }
    }
}

fn render_at_picker(f: &mut Frame, app: &AppState, area: Rect) {
    if area.width < 10 || area.height < 3 {
        return;
    }
    let focused = app.focus == AppFocus::Input;
    let visible_rows = (area.height as usize).saturating_sub(2).max(1);
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(
            "@ pick (cwd): ",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(truncate_line(
            &app.at_query,
            area.width.saturating_sub(14) as usize,
        )),
    ]));

    if app.at_results.is_empty() {
        lines.push(Line::from(Span::raw("(no matches)")));
    } else {
        let total = app.at_results.len();
        let scroll_start = if visible_rows >= total {
            0
        } else {
            app.at_scroll.min(total.saturating_sub(visible_rows))
        };
        let end = (scroll_start + visible_rows).min(total);
        for (idx, it) in app.at_results[scroll_start..end].iter().enumerate() {
            let selected = idx == 0; // marker fixed to top visible row
            let mut label = it.rel.clone();
            if it.is_dir && !label.ends_with('/') {
                label.push('/');
            }
            lines.push(Line::from(Span::styled(
                truncate_line(
                    &format!("{} {}", if selected { "›" } else { " " }, label),
                    area.width.saturating_sub(1) as usize,
                ),
                if selected {
                    Style::default().add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                },
            )));
        }
    }

    let b = panel(app, " @ ", focused);
    let p = Paragraph::new(lines).block(b).wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn render_inspector(f: &mut Frame, app: &AppState, area: Rect) {
    let focused = app.focus == AppFocus::Inspector;

    let titles = vec!["Status", "Tasks", "Team", "Events", "Tools"];
    let idx = match app.inspector_tab {
        InspectorTab::Status => 0,
        InspectorTab::Tasks => 1,
        InspectorTab::Team => 2,
        InspectorTab::Events => 3,
        InspectorTab::Tools => 4,
    };
    let tabs = Tabs::new(titles)
        .select(idx)
        .block(panel(app, " inspector ", focused))
        .highlight_style(
            Style::default()
                .fg(app.tokens.accent)
                .add_modifier(Modifier::BOLD),
        );
    f.render_widget(tabs, area);

    let inner = Rect {
        x: area.x + 1,
        y: area.y + 2,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(3),
    };

    let body = match app.inspector_tab {
        InspectorTab::Status => render_status_lines(app, inner.width as usize),
        InspectorTab::Tasks => render_text_block(&app.cached_task_list, inner.width as usize),
        InspectorTab::Team => render_text_block(&app.cached_team_list, inner.width as usize),
        InspectorTab::Events => render_text_block(&app.cached_events, inner.width as usize),
        InspectorTab::Tools => render_tools_lines(app, inner.width as usize),
    };

    let para = Paragraph::new(body).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

fn render_status_lines(app: &AppState, width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let tokens = app.cached_tokens;
    let turns = app.cached_turns;
    let uptime = app.started_at.elapsed();
    let mins = uptime.as_secs() / 60;
    let secs = uptime.as_secs() % 60;

    lines.push(Line::from(Span::styled(
        truncate_line(&format!("backend: {} ({})", app.backend, app.model), width),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(Span::raw(truncate_line(
        &format!("agent:   {} [{}]", app.agent_name, app.agent_role),
        width,
    ))));
    lines.push(Line::from(Span::raw(truncate_line(
        &format!("session: {}", app.session_id),
        width,
    ))));
    lines.push(Line::from(Span::raw(truncate_line(
        &format!("turns:   {}", turns),
        width,
    ))));
    lines.push(Line::from(Span::raw(truncate_line(
        &format!("tokens:  ~{}", tokens),
        width,
    ))));
    lines.push(Line::from(Span::raw(truncate_line(
        &format!("uptime:  {}m {}s", mins, secs),
        width,
    ))));

    if let Some(w) = app.last_warning.as_ref() {
        lines.push(Line::from(Span::raw(truncate_line(
            &format!("last warning: {}", w),
            width,
        ))));
    }
    if let Some(e) = app.last_error.as_ref() {
        lines.push(Line::from(Span::raw(truncate_line(
            &format!("last error: {}", e),
            width,
        ))));
    }

    if !app.last_at_refs.is_empty() {
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "@refs",
            Style::default().add_modifier(Modifier::BOLD),
        )));
        for r in app.last_at_refs.iter().take(8) {
            let kind = match r.kind {
                super::app_state::AtRefKind::File => "file",
                super::app_state::AtRefKind::Dir => "dir",
                super::app_state::AtRefKind::Missing => "missing",
            };
            lines.push(Line::from(Span::raw(truncate_line(
                &format!("- [{}] {} → {}", kind, r.display, r.resolved),
                width,
            ))));
        }
        if app.last_at_refs.len() > 8 {
            lines.push(Line::from(Span::raw(truncate_line(
                &format!("… and {} more", app.last_at_refs.len().saturating_sub(8)),
                width,
            ))));
        }
    }
    lines.push(Line::from(Span::raw("")));

    lines.push(Line::from(Span::styled(
        "Todo",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    lines.extend(render_text_block(&app.cached_todo_state, width));

    lines.push(Line::from(Span::raw("")));
    lines.push(Line::from(Span::raw("Tip: /status /tasks /team /events")));
    lines
}

fn render_tools_lines(app: &AppState, width: usize) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    out.push(Line::from(Span::styled(
        "Activity (tool/subagent/background/resilience)",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    out.push(Line::from(Span::raw("")));

    let mut shown = 0usize;
    let now = std::time::Instant::now();
    for item in app.activity.iter().rev() {
        let age_ms = now.duration_since(item.at).as_millis();
        let age = if age_ms >= 60_000 {
            format!("{}m", age_ms / 60_000)
        } else if age_ms >= 1000 {
            format!("{}s", age_ms / 1000)
        } else {
            format!("{}ms", age_ms)
        };
        let lvl = match item.level {
            super::events::LogLevel::Debug => "dbg",
            super::events::LogLevel::Info => "info",
            super::events::LogLevel::Warn => "warn",
            super::events::LogLevel::Error => "err",
        };
        out.push(Line::from(Span::raw(truncate_line(
            &format!("[{}] {} {} — {}", age, item.source, lvl, item.message),
            width,
        ))));
        shown += 1;
        if shown >= 20 {
            break;
        }
    }

    if shown == 0 {
        out.push(Line::from(Span::raw("No activity yet.")));
    }
    out
}

fn render_text_block(text: &str, width: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for line in text.lines() {
        out.push(Line::from(Span::raw(truncate_line(line, width))));
    }
    if out.is_empty() {
        out.push(Line::from(Span::raw("(empty)")));
    }
    out
}

fn render_toasts(f: &mut Frame, app: &AppState, area: Rect) {
    if app.toasts.is_empty() {
        return;
    }
    let msg = app
        .toasts
        .iter()
        .map(|(m, _)| m.clone())
        .collect::<Vec<_>>()
        .join("  •  ");
    if area.width == 0 {
        return;
    }
    // Render inside a 1-line box aligned to the top-right.
    // IMPORTANT: keep x + width <= area.x + area.width (ratatui will panic otherwise).
    let content_w = msg.width().min(area.width.saturating_sub(2) as usize);
    let box_w = (content_w as u16).saturating_add(2).min(area.width);
    let x = area.x + area.width.saturating_sub(box_w);
    let r = Rect {
        x,
        y: area.y,
        width: box_w,
        height: 1,
    };
    let p = Paragraph::new(msg)
        .style(
            Style::default()
                .fg(app.tokens.accent)
                .add_modifier(Modifier::BOLD),
        )
        .alignment(Alignment::Right);
    f.render_widget(p, r);
}

fn render_help(f: &mut Frame, _app: &AppState, area: Rect) {
    if area.width < 10 || area.height < 6 {
        return;
    }
    let w = area.width.min(70);
    let h = area.height.min(18);
    let r = centered(area, w, h);
    let lines = vec![
        Line::from(Span::styled(
            "Help",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::raw("")),
        Line::from(Span::raw("Ctrl+Q or /quit   quit TUI")),
        Line::from(Span::raw("Ctrl+C   interrupt agent turn / clear input")),
        Line::from(Span::raw("Tab      cycle focus")),
        Line::from(Span::raw("PageUp/Down scroll chat")),
        Line::from(Span::raw("Shift+Enter newline")),
        Line::from(Span::raw("Enter    send")),
        Line::from(Span::raw("")),
        Line::from(Span::raw("Palette:")),
        Line::from(Span::raw(
            "/   session/UI — open when input empty; or type /cmd and Enter.",
        )),
        Line::from(Span::raw(
            "    status, tasks, team, events, help, clear, quit",
        )),
        Line::from(Span::raw(
            ":   panels/toggles — open any time. status, tasks, team, events,",
        )),
        Line::from(Span::raw("    tools, inspector (toggle), sidebar (toggle)")),
        Line::from(Span::raw("Esc      close overlays")),
    ];
    let b = Block::default().title(" help ").borders(Borders::ALL);
    let p = Paragraph::new(lines).block(b).wrap(Wrap { trim: false });
    f.render_widget(Clear, r);
    f.render_widget(p, r);
}

fn render_palette(f: &mut Frame, _app: &AppState, palette: &CommandPalette, area: Rect) {
    if area.width < 10 || area.height < 6 {
        return;
    }
    let w = area.width.min(60);
    let h = area.height.clamp(7, 12);
    let r = centered(area, w, h);
    let title = match palette.mode {
        PaletteMode::Slash => " / session ",
        PaletteMode::Colon => " : panels ",
        PaletteMode::None => " ",
    };
    let b = Block::default().title(title).borders(Borders::ALL);
    let items = palette.filtered_items();
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("query: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(palette.query().to_string()),
    ]));
    lines.push(Line::from(Span::raw("")));
    for (i, it) in items
        .iter()
        .take((h as usize).saturating_sub(4))
        .enumerate()
    {
        let selected = i == palette.selection();
        lines.push(Line::from(Span::styled(
            format!("{} {}", if selected { "›" } else { " " }, it),
            if selected {
                Style::default().add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            },
        )));
    }
    if items.is_empty() {
        lines.push(Line::from(Span::raw("(no matches)")));
    }
    let p = Paragraph::new(lines).block(b);
    f.render_widget(Clear, r);
    f.render_widget(p, r);
}

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect {
        x,
        y,
        width: w,
        height: h,
    }
}

fn truncate_line(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.width() <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut w = 0usize;
    for ch in s.chars() {
        let cw = ch.to_string().width();
        if w + cw >= max.saturating_sub(1) {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('…');
    out
}

fn wrap_text(s: &str, max: usize) -> Vec<String> {
    if max == 0 {
        return vec![String::new()];
    }
    if s.is_empty() {
        return vec![String::new()];
    }

    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0usize;

    // Split by whitespace, but keep it simple and predictable for terminal rendering.
    for word in s.split_whitespace() {
        let word_w = word.width();

        // If the word itself doesn't fit on an empty line, hard-wrap by characters.
        if cur.is_empty() && word_w > max {
            let mut chunk = String::new();
            let mut w = 0usize;
            for ch in word.chars() {
                let cw = ch.to_string().width();
                if w + cw > max && !chunk.is_empty() {
                    out.push(chunk);
                    chunk = String::new();
                    w = 0;
                }
                chunk.push(ch);
                w += cw;
            }
            if !chunk.is_empty() {
                out.push(chunk);
            }
            cur.clear();
            cur_w = 0;
            continue;
        }

        let sep_w = if cur.is_empty() { 0 } else { 1 };
        if cur_w + sep_w + word_w <= max {
            if sep_w == 1 {
                cur.push(' ');
            }
            cur.push_str(word);
            cur_w += sep_w + word_w;
        } else {
            if !cur.is_empty() {
                out.push(cur);
            }
            cur = word.to_string();
            cur_w = word_w.min(max);
        }
    }

    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn wrap_preserve_whitespace(s: &str, max: usize) -> Vec<String> {
    if max == 0 {
        return vec![String::new()];
    }
    if s.is_empty() {
        return vec![String::new()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0usize;
    for ch in s.chars() {
        let cw = ch.to_string().width();
        if cur_w + cw > max && !cur.is_empty() {
            out.push(cur);
            cur = String::new();
            cur_w = 0;
        }
        cur.push(ch);
        cur_w += cw;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn format_message_block(label: &str, text: &str, width: usize) -> Vec<Line<'static>> {
    let label_span = Span::styled(
        label.to_string(),
        Style::default().add_modifier(Modifier::BOLD),
    );
    let mut out: Vec<Line<'static>> = Vec::new();

    let mut in_code = false;
    let mut label_needed = true;
    for raw in text.lines() {
        let line = raw.trim_end_matches('\r');
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            if in_code {
                out.push(Line::from(vec![
                    if label_needed {
                        label_needed = false;
                        label_span.clone()
                    } else {
                        Span::raw("".to_string())
                    },
                    Span::raw("  "),
                    Span::styled("┌─ code", Style::default().add_modifier(Modifier::BOLD)),
                ]));
            } else {
                out.push(Line::from(vec![
                    Span::raw(""),
                    Span::raw("  "),
                    Span::raw("└─"),
                ]));
            }
            continue;
        }

        if in_code {
            let avail = width.saturating_sub(6).saturating_sub(2); // account for "│ "
            for (i, part) in wrap_preserve_whitespace(line, avail)
                .into_iter()
                .enumerate()
            {
                out.push(Line::from(vec![
                    if label_needed {
                        label_needed = false;
                        label_span.clone()
                    } else {
                        Span::raw("".to_string())
                    },
                    Span::raw("  "),
                    Span::styled(
                        truncate_line(&format!("│ {}", part), width.saturating_sub(6)),
                        Style::default().add_modifier(Modifier::DIM),
                    ),
                ]));
                // For subsequent wrapped lines inside code block, keep label blank.
                if i == 0 {
                    // label_needed already flipped above when needed
                }
            }
            continue;
        }

        let avail = width.saturating_sub(4);
        for part in wrap_text(line, avail) {
            out.push(Line::from(vec![
                if label_needed {
                    label_needed = false;
                    label_span.clone()
                } else {
                    Span::raw("".to_string())
                },
                Span::raw("  "),
                Span::raw(part),
            ]));
        }
    }

    if out.is_empty() {
        out.push(Line::from(vec![label_span, Span::raw("  "), Span::raw("")]));
    }
    out
}

fn format_tool_card_start(name: &str, preview: &str, width: usize) -> Vec<Line<'static>> {
    let title = format!("tool: {}", name);
    let mut out = Vec::new();
    out.push(Line::from(vec![
        Span::styled("┌", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(" {}", title),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ]));

    let avail = width.saturating_sub(2).saturating_sub(2); // account for "│ "
    for part in wrap_text(preview, avail) {
        out.push(Line::from(vec![Span::styled(
            truncate_line(&format!("│ {}", part), width.saturating_sub(2)),
            Style::default().add_modifier(Modifier::DIM),
        )]));
    }
    out.push(Line::from(vec![Span::raw("└")]));
    out
}

fn format_tool_card_done(
    ok: bool,
    msg: &str,
    seconds: f64,
    width: usize,
    ok_color: Color,
    err_color: Color,
) -> Vec<Line<'static>> {
    let (tag, color) = if ok {
        ("ok", ok_color)
    } else {
        ("err", err_color)
    };
    let headline = format!("{} {:.1}s", tag, seconds);
    let mut out = Vec::new();
    out.push(Line::from(vec![
        Span::styled("┌", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(" {}", headline),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
    ]));

    let avail = width.saturating_sub(2).saturating_sub(2); // account for "│ "
    for part in wrap_text(msg, avail) {
        out.push(Line::from(vec![Span::raw(truncate_line(
            &format!("│ {}", part),
            width.saturating_sub(2),
        ))]));
    }
    out.push(Line::from(vec![Span::raw("└")]));
    out
}

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::app_state::{AppState, InspectorTab};

#[derive(Default, Clone)]
pub struct InputBuffer {
    text: String,
    cursor: usize, // byte index
}

impl InputBuffer {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    pub fn insert_char(&mut self, ch: char) {
        self.text.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    /// Insert a string at the cursor (e.g. for pasted text). Supports large pastes.
    pub fn insert_str(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn set_cursor(&mut self, cursor: usize) {
        self.cursor = cursor.min(self.text.len());
    }

    pub fn replace_range(&mut self, start: usize, end: usize, replacement: &str) {
        let s = start.min(self.text.len());
        let e = end.min(self.text.len());
        let (s, e) = if s <= e { (s, e) } else { (e, s) };
        self.text.replace_range(s..e, replacement);
        self.cursor = s + replacement.len();
    }

    pub fn newline(&mut self) {
        self.insert_char('\n');
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.text[..self.cursor]
            .char_indices()
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.text.remove(prev);
        self.cursor = prev;
    }

    pub fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor = self.text[..self.cursor]
            .char_indices()
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    pub fn move_right(&mut self) {
        if self.cursor >= self.text.len() {
            return;
        }
        let next = self.text[self.cursor..]
            .char_indices()
            .nth(1)
            .map(|(i, _)| self.cursor + i)
            .unwrap_or(self.text.len());
        self.cursor = next;
    }

    pub fn take(&mut self) -> String {
        let t = self.text.clone();
        self.clear();
        t
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PaletteMode {
    None,
    Slash,
    Colon,
}

pub struct CommandPalette {
    pub mode: PaletteMode,
    query: String,
    selection: usize,
}

impl Default for CommandPalette {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandPalette {
    pub fn new() -> Self {
        Self {
            mode: PaletteMode::None,
            query: String::new(),
            selection: 0,
        }
    }

    pub fn is_open(&self) -> bool {
        self.mode != PaletteMode::None
    }

    pub fn open_slash(&mut self) {
        self.mode = PaletteMode::Slash;
        self.query.clear();
        self.selection = 0;
    }

    pub fn open_colon(&mut self) {
        self.mode = PaletteMode::Colon;
        self.query.clear();
        self.selection = 0;
    }

    pub fn close(&mut self) {
        self.mode = PaletteMode::None;
        self.query.clear();
        self.selection = 0;
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn selection(&self) -> usize {
        self.selection
    }

    pub fn items(&self) -> Vec<&'static str> {
        match self.mode {
            PaletteMode::Slash => {
                vec!["status", "tasks", "team", "events", "help", "clear", "quit"]
            }
            PaletteMode::Colon => vec![
                "status",
                "tasks",
                "team",
                "events",
                "tools",
                "inspector",
                "sidebar",
            ],
            PaletteMode::None => vec![],
        }
    }

    pub fn filtered_items(&self) -> Vec<&'static str> {
        let q = self.query.trim().to_lowercase();
        let items = self.items();
        if q.is_empty() {
            return items;
        }
        items
            .into_iter()
            .filter(|it| it.to_lowercase().contains(&q))
            .collect()
    }

    /// Returns true if event handled.
    pub fn handle_key(&mut self, key: KeyEvent, app: &mut AppState) -> bool {
        match key {
            KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                self.close();
                true
            }
            KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                let items = self.filtered_items();
                if items.is_empty() {
                    return true;
                }
                if self.selection == 0 {
                    self.selection = items.len() - 1;
                } else {
                    self.selection -= 1;
                }
                true
            }
            KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                let items = self.filtered_items();
                if items.is_empty() {
                    return true;
                }
                self.selection = (self.selection + 1) % items.len();
                true
            }
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                self.query.pop();
                self.selection = 0;
                true
            }
            KeyEvent {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                let items = self.filtered_items();
                let q = if !items.is_empty() {
                    items[self.selection.min(items.len().saturating_sub(1))]
                } else {
                    self.query.trim()
                };
                if self.mode == PaletteMode::Slash {
                    let cmd = format!("/{}", q);
                    app.handle_slash_command(&cmd);
                } else if self.mode == PaletteMode::Colon {
                    match q {
                        "status" => {
                            app.inspector_tab = InspectorTab::Status;
                            app.inspector_visible = true;
                        }
                        "tasks" => {
                            app.inspector_tab = InspectorTab::Tasks;
                            app.inspector_visible = true;
                        }
                        "team" => {
                            app.inspector_tab = InspectorTab::Team;
                            app.inspector_visible = true;
                        }
                        "events" => {
                            app.inspector_tab = InspectorTab::Events;
                            app.inspector_visible = true;
                        }
                        "tools" => {
                            app.inspector_tab = InspectorTab::Tools;
                            app.inspector_visible = true;
                        }
                        "inspector" => app.inspector_visible = !app.inspector_visible,
                        "sidebar" => app.sidebar_visible = !app.sidebar_visible,
                        _ => app.push_toast("Unknown command"),
                    }
                }
                self.close();
                true
            }
            KeyEvent {
                code: KeyCode::Char(ch),
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                self.query.push(ch);
                self.selection = 0;
                true
            }
            _ => false,
        }
    }
}

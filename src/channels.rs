use std::sync::{Arc, Mutex};

use nu_ansi_term::{Color, Style};
use reedline::{
    default_emacs_keybindings, ColumnarMenu, DefaultHinter, Emacs, FileBackedHistory, KeyCode,
    KeyModifiers, MenuBuilder, Reedline, ReedlineEvent, ReedlineMenu, Signal,
};

use crate::ui::{NanoHighlighter, NanoPrompt, NanoValidator, SlashCompleter};

// ---------------------------------------------------------------------------
// Inbound message from any channel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct InboundMessage {
    pub text: String,
    pub sender_id: String,
    pub channel: String,
    pub peer_id: String,
    pub is_group: bool,
    pub media: Vec<MediaAttachment>,
    pub raw: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct MediaAttachment {
    pub kind: String, // "image", "file", "audio", etc.
    pub url: String,
    pub name: String,
}

// ---------------------------------------------------------------------------
// Channel trait
// ---------------------------------------------------------------------------

pub trait Channel: Send + Sync {
    fn name(&self) -> &str;
    fn recv(&self) -> Option<InboundMessage>;
    fn send(&self, peer_id: &str, text: &str) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// CLI Channel: reedline-based interactive input
// ---------------------------------------------------------------------------

pub struct CliChannel {
    editor: Mutex<Reedline>,
    prompt: NanoPrompt,
}

impl CliChannel {
    pub fn new(
        prompt: NanoPrompt,
        history_path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let commands: Vec<String> = vec![
            "/quit", "/clear", "/status", "/tasks", "/team", "/events", "/resume", "/help",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let completer = Box::new(SlashCompleter::new(commands));

        let hinter = Box::new(
            DefaultHinter::default().with_style(Style::new().dimmed().fg(Color::DarkGray)),
        );

        let highlighter = Box::new(NanoHighlighter);
        let validator = Box::new(NanoValidator);

        let history = Box::new(FileBackedHistory::with_file(
            1000,
            history_path.to_path_buf(),
        )?);

        let completion_menu = Box::new(ColumnarMenu::default().with_name("completion_menu"));

        let mut keybindings = default_emacs_keybindings();
        keybindings.add_binding(
            KeyModifiers::NONE,
            KeyCode::Tab,
            ReedlineEvent::UntilFound(vec![
                ReedlineEvent::Menu("completion_menu".to_string()),
                ReedlineEvent::MenuNext,
            ]),
        );
        let edit_mode = Box::new(Emacs::new(keybindings));

        let editor = Reedline::create()
            .with_history(history)
            .with_completer(completer)
            .with_hinter(hinter)
            .with_highlighter(highlighter)
            .with_validator(validator)
            .with_menu(ReedlineMenu::EngineCompleter(completion_menu))
            .with_edit_mode(edit_mode);

        Ok(Self {
            editor: Mutex::new(editor),
            prompt,
        })
    }

    /// Blocking read from the terminal. Returns `Some(text)` on success,
    /// `None` on Ctrl-C, or `Some("/quit")` on Ctrl-D / EOF.
    pub fn read_line(&self) -> Option<String> {
        let mut editor = self.editor.lock().unwrap();
        match editor.read_line(&self.prompt) {
            Ok(Signal::Success(line)) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            }
            Ok(Signal::CtrlC) => None,
            Ok(Signal::CtrlD) => Some("/quit".to_string()),
            Err(_) => Some("/quit".to_string()),
        }
    }

    /// Sync history to disk.
    pub fn sync_history(&self) {
        if let Ok(mut editor) = self.editor.lock() {
            let _ = editor.sync_history();
        }
    }
}

impl Channel for CliChannel {
    fn name(&self) -> &str {
        "cli"
    }

    fn recv(&self) -> Option<InboundMessage> {
        // In the new architecture, read_line() is called directly from the main loop.
        // Channel::recv() is not used for CLI input.
        None
    }

    fn send(&self, _peer_id: &str, text: &str) -> Result<(), String> {
        // Used by DeliveryRunner to send messages to the CLI channel
        println!(
            "\n {}\u{25C6}{} {}",
            Style::new().dimmed().fg(Color::Cyan).prefix(),
            Style::new().dimmed().fg(Color::Cyan).suffix(),
            text
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Channel Manager
// ---------------------------------------------------------------------------

pub struct ChannelManager {
    channels: Vec<Arc<dyn Channel>>,
}

impl Default for ChannelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelManager {
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
        }
    }

    pub fn add(&mut self, channel: Arc<dyn Channel>) {
        self.channels.push(channel);
    }

    /// Poll all channels, returning the first available message.
    pub fn poll(&self) -> Option<InboundMessage> {
        for channel in &self.channels {
            if let Some(msg) = channel.recv() {
                return Some(msg);
            }
        }
        None
    }

    /// Send a message to a specific peer on a named channel.
    pub fn send(&self, channel_name: &str, peer_id: &str, text: &str) -> Result<(), String> {
        for channel in &self.channels {
            if channel.name() == channel_name {
                return channel.send(peer_id, text);
            }
        }
        Err(format!("Channel '{}' not found", channel_name))
    }

    /// Get channel names.
    pub fn channel_names(&self) -> Vec<String> {
        self.channels.iter().map(|c| c.name().to_string()).collect()
    }
}

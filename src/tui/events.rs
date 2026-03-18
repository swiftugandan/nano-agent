use std::sync::mpsc;

use crate::types::ToolEvent;

#[derive(Clone, Debug)]
pub struct UiEventSender(mpsc::Sender<UiEvent>);

impl UiEventSender {
    pub fn new(inner: mpsc::Sender<UiEvent>) -> Self {
        Self(inner)
    }

    pub fn send(&self, evt: UiEvent) -> Result<(), mpsc::SendError<UiEvent>> {
        self.0.send(evt)
    }
}

#[derive(Debug)]
pub enum UiEvent {
    Tool(ToolEvent),
    Warning(String),
    Error(String),
    Toast(String),
    Background {
        source: String,
        message: String,
    },
    BusEvent {
        event: String,
        data: serde_json::Value,
    },
    SubagentProgress {
        message: String,
    },
    LogLine {
        source: String,
        level: LogLevel,
        message: String,
    },
    AgentFinished {
        assistant_text: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

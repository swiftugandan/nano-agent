use clap::Parser;

#[derive(Parser)]
#[command(name = "agent", version, about = "Autonomous coding agent")]
pub struct Cli {
    /// Prompt to send (enables one-shot mode)
    pub prompt: Option<String>,

    /// Resume a previous session
    #[arg(long)]
    pub resume: Option<String>,

    /// Start WebSocket gateway on this address
    #[arg(long)]
    pub gateway: Option<String>,

    /// Force full-screen TUI mode (interactive only)
    #[arg(long)]
    pub tui: bool,

    /// Force legacy REPL mode (interactive only)
    #[arg(long)]
    pub repl: bool,

    /// Disable ANSI colors (TUI + REPL)
    #[arg(long)]
    pub no_color: bool,

    /// Theme name for TUI (currently: bold, hacker)
    #[arg(long, default_value = "bold")]
    pub theme: String,
}

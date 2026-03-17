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
}

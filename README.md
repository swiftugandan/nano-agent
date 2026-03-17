# nano-agent

A Rust-based autonomous coding agent that runs an interactive loop with an LLM, using tools for file I/O, planning, tasks, background execution, team messaging, subagent delegation, and git worktree isolation.

**Code index:** [DeepWiki — swiftugandan/nano-agent](https://deepwiki.com/swiftugandan/nano-agent)  
**Original inspiration:** [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) — *Bash is all you need: a nano Claude Code–like agent, built from 0 to 1*

## Requirements

- **Rust** (2021 edition)
- **LLM API keys** (one of):
  - Anthropic: `ANTHROPIC_API_KEY` (default backend)
  - OpenRouter: `OPENROUTER_API_KEY` — set `LLM_BACKEND=openai` to use

## Quick Start

```bash
# 1. Build
cargo build --release

# 2. Set an API key (pick one)
export ANTHROPIC_API_KEY="sk-ant-..."          # default backend
# — or —
export LLM_BACKEND=openai
export OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_MODEL="openai/gpt-4o-mini"   # any OpenRouter model

# 3. Run
./target/release/agent
```

### Usage

**Interactive (REPL)** — launch with no arguments:

```bash
agent
```

The REPL provides readline-style editing (arrow keys, Ctrl-A/E/K), persistent history (up arrow recalls previous inputs across sessions), and tab-completion for slash commands (type `/` then Tab).

- **Ctrl-C** — cancel current input, show new prompt
- **Ctrl-D** — clean exit (same as `/quit`)

**One-shot** — pass a prompt as an argument:

```bash
agent "explain src/main.rs in one paragraph"
```

Prints the response to stdout and exits. Useful for scripting and pipes:

```bash
cat error.log | agent "what went wrong?"
agent "list TODOs in src/" > todos.txt
```

**Resume a session:**

```bash
agent --resume <session-id>
```

Transcripts and history are stored under `.nano-agent/` in the current working directory.

## Environment

| Variable       | Default      | Description                          |
|----------------|--------------|--------------------------------------|
| `LLM_BACKEND`  | `anthropic`  | `anthropic` or `openai`              |
| `AGENT_NAME`   | `lead`       | Agent identity name                  |
| `AGENT_ROLE`   | `developer`  | Agent role in system prompt          |
| `SKILLS_DIR`   | `.nano-agent/skills` | Directory for skill files (see `read_skill`) |
| `OPENROUTER_API_KEY` | —     | API key for OpenRouter backend       |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Model to use via OpenRouter |
| `ANTHROPIC_API_KEY_2..10` | — | Additional API keys for rotation on auth errors |
| `RETRY_MAX_ATTEMPTS` | `3`   | Max retry attempts for transient LLM errors |
| `RETRY_BASE_DELAY_MS` | `1000` | Base delay (ms) for exponential backoff |
| `RESUME_SESSION` | —         | Session ID to resume on startup      |

Data is stored under **`.nano-agent/`** in the current working directory: `tasks`, `transcripts`, `inbox`, `team`, `events.jsonl`, and `skills`.

## Commands

In the REPL you can use:

- **`/quit`** — Exit and save transcript
- **`/clear`** — Save transcript, clear conversation, start fresh
- **`/status`** — Show token estimate, turn count, todo list, background completions
- **`/tasks`** — List all tasks (task manager)
- **`/team`** — List teammates
- **`/events`** — Show recent event-bus entries
- **`/resume`** — List available sessions for resumption

## Capabilities (tools)

The agent has many tools, including:

- **File I/O** — read, write, search, list files
- **Planning** — `todo_update`, `todo_read` (single in-progress item)
- **Tasks** — `task_create`, `task_get`, `task_update`, `task_list` (with dependencies)
- **Background** — `background_run`, `background_check`
- **Team** — `send_message`, `broadcast_message`, `scan_tasks`, `claim_task`
- **Delegation** — `subagent` (isolated subtask with child LLM call)
- **Isolation** — `worktree_create` (git worktree per task)
- **Knowledge** — `read_skill` (load skills from `SKILLS_DIR`)
- **Events** — `emit_event`, `list_events`
- **Memory** — `save_memory` (persist context for TF-IDF recall)
- **Lifecycle** — `idle` (transition to idle), `compact` (manual compaction)
- **Cron** — `cron_add`, `cron_list`, `cron_remove`
- **Team** — `spawn_teammate`, `shutdown_teammate`
- **Delivery** — `enqueue_delivery` (reliable message delivery with retry)

The system prompt instructs the agent to keep one todo in progress, use tasks for multi-step work, use background runs for long commands, and coordinate via the message bus and task claiming.

## Development

### Building

```bash
cargo build              # debug build
cargo build --release    # optimized build
```

### Running Tests

```bash
cargo test               # run all tests
cargo test test_l1       # run a specific test suite (e.g. core loop)
cargo test scenario_19   # run a single scenario
```

### Adding a Tool

1. Define the tool schema in `src/tools.rs` (`tool_definitions()`)
2. Add a handler closure in `src/main.rs` (`build_dispatch()`)
3. Add tests in the appropriate `tests/` file

### Adding a Slash Command

Slash commands are handled in the REPL loop in `src/main.rs`. To add tab-completion for a new command, add it to the `SlashCompleter` command list in `src/channels.rs`.

### Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full 11-layer design. The core invariant is a simple loop: call the LLM, append its response, execute any tool calls, append results, repeat until the LLM stops.

## Project Layout

- **`src/main.rs`** — Binary: REPL, tool wiring, system prompt, pre-turn pipeline (memory, nag, background, inbox).
- **`src/lib.rs`** — Library root; modules:
  - **LLM** — `anthropic`, `openai`, `types` (e.g. `Llm` trait)
  - **Loop** — `core_loop` (run agent turn with tools)
  - **Tools** — `tools` (definitions + dispatch), `delegation` (subagent)
  - **Planning** — `planning` (todo, nag policy)
  - **Tasks** — `tasks` (TaskManager, dependency graph)
  - **Memory** — `memory` (compaction, transcript store, session replay), `memory_store` (TF-IDF semantic recall)
  - **Concurrency** — `concurrency` (background runner)
  - **Teams** — `teams` (message bus, teammates)
  - **Protocols** — `protocols` (e.g. request tracking)
  - **Autonomy** — `autonomy` (identity, task scan/claim)
  - **Isolation** — `isolation` (event bus, worktree manager)
  - **Knowledge** — `knowledge` (skill loader)
  - **Channels** — `channels` (Channel trait, CliChannel, ChannelManager), `gateway` (WebSocket server, routing)
  - **Delivery** — `delivery` (write-ahead queue, background retry runner)
  - **Resilience** — `resilience` (ResilientLlm, ContextGuard, retry policy)
  - **Prompt** — `prompt` (PromptAssembler, layered composition)
  - **Util** — `util` (shared helpers)
  - **Mock** — `mock` (test doubles for LLM)

## License

See repository license (if any).

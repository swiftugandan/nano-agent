# nano-agent

A Rust-based autonomous coding agent that runs an interactive loop with an LLM, using tools for file I/O, planning, tasks, background execution, team messaging, subagent delegation, and git worktree isolation.

**Code index:** [DeepWiki — swiftugandan/nano-agent](https://deepwiki.com/swiftugandan/nano-agent)  
**Original inspiration:** [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) — *Bash is all you need: a nano Claude Code–like agent, built from 0 to 1*

## Requirements

- **Rust** (2021 edition)
- **LLM API keys** (one of):
  - Anthropic: `ANTHROPIC_API_KEY` (default backend)
  - OpenAI: `OPENAI_API_KEY` — set `LLM_BACKEND=openai` to use

## Build & Run

```bash
cargo build --release
./target/release/agent
```

Or from the project root:

```bash
cargo run --bin agent
```

The agent starts an interactive prompt (`> `). Type your request and press Enter; the agent uses the configured LLM and tools to respond. Transcripts are saved under `.nano-agent/transcripts` on exit or `/clear`.

## Environment

| Variable       | Default      | Description                          |
|----------------|--------------|--------------------------------------|
| `LLM_BACKEND`  | `anthropic`  | `anthropic` or `openai`              |
| `AGENT_NAME`   | `lead`       | Agent identity name                  |
| `AGENT_ROLE`   | `developer`  | Agent role in system prompt          |
| `SKILLS_DIR`   | `.nano-agent/skills` | Directory for skill files (see `read_skill`) |

Data is stored under **`.nano-agent/`** in the current working directory: `tasks`, `transcripts`, `inbox`, `team`, `events.jsonl`, and `skills`.

## Commands

In the REPL you can use:

- **`/quit`** — Exit and save transcript
- **`/clear`** — Save transcript, clear conversation, start fresh
- **`/status`** — Show token estimate, turn count, todo list, background completions
- **`/tasks`** — List all tasks (task manager)
- **`/team`** — List teammates
- **`/events`** — Show recent event-bus entries

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

The system prompt instructs the agent to keep one todo in progress, use tasks for multi-step work, use background runs for long commands, and coordinate via the message bus and task claiming.

## Tests

```bash
cargo test
```

Integration tests cover the core loop, planning, delegation, knowledge, memory, tasks, concurrency, teams, protocols, autonomy, isolation, and LLM backends (with mocks where needed).

## Project layout

- **`src/main.rs`** — Binary: REPL, tool wiring, system prompt, pre-turn pipeline (memory, nag, background, inbox).
- **`src/lib.rs`** — Library root; modules:
  - **LLM** — `anthropic`, `openai`, `types` (e.g. `Llm` trait)
  - **Loop** — `core_loop` (run agent turn with tools)
  - **Tools** — `tools` (definitions + dispatch), `delegation` (subagent)
  - **Planning** — `planning` (todo, nag policy)
  - **Tasks** — `tasks` (TaskManager, dependency graph)
  - **Memory** — `memory` (compaction, transcript store)
  - **Concurrency** — `concurrency` (background runner)
  - **Teams** — `teams` (message bus, teammates)
  - **Protocols** — `protocols` (e.g. request tracking)
  - **Autonomy** — `autonomy` (identity, task scan/claim)
  - **Isolation** — `isolation` (event bus, worktree manager)
  - **Knowledge** — `knowledge` (skill loader)
  - **Mock** — `mock` (test doubles for LLM)

## License

See repository license (if any).

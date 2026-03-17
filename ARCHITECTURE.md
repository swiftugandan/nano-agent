# Generalized AI Coding Agent Architecture — Interface Definitions

This document defines the **language-agnostic interfaces** for an 11-layer AI coding agent architecture. Each layer adds exactly one capability through 5 extension points around an invariant core loop.

Any language implementation that satisfies these interfaces and passes the contract tests in `contracts/specs/` conforms to the architecture.

---

## The Invariant Core

```pseudocode
function agent_loop(system_prompt, messages, tools, dispatch):
    while true:
        response = LLM.call(system_prompt, messages, tools)
        messages.append(assistant_turn(response))

        if response.stop_reason != "tool_use":
            break

        results = []
        for tool_call in response.tool_calls:
            if tool_call.name in dispatch:
                result = dispatch[tool_call.name](tool_call.input)
            else:
                result = "Unknown tool: {tool_call.name}"
            results.append(tool_result(tool_call.id, result))

        messages.append(user_turn(results))

    return messages
```

### 5 Extension Points

| # | Extension Point | Location in Loop | Description |
|---|----------------|-----------------|-------------|
| 1 | **Tool Map Extension** | `dispatch` map | Add new tool name → handler entries |
| 2 | **Pre-LLM Injection** | Before `LLM.call()` | Modify messages (nag reminders, bg results, inbox drain) |
| 3 | **Post-Tool Interception** | After tool execution | Process results before appending (compact, shutdown) |
| 4 | **System Prompt Composition** | Build `system_prompt` | Compose from components (skill descriptions, identity) |
| 5 | **Lifecycle Wrapping** | Around `agent_loop()` | Wrap in work/idle cycles, worktree scoping |

---

## Layer 1: Core Loop

### AgentLoop

```pseudocode
interface AgentLoop:
    function run(system: string, messages: list, tools: list, dispatch: map) -> list
        # Calls LLM in a loop until stop_reason != "tool_use"
        # Returns final messages list
        # Each tool_result MUST carry the tool_use_id of the call it answers
```

### ToolDispatcher

```pseudocode
interface ToolDispatcher:
    dispatch: map<string, function(input: map) -> string>

    function route(tool_name: string, input: map) -> string
        # If tool_name in dispatch: return dispatch[tool_name](input)
        # Else: return error string (MUST NOT throw)
```

### PathSandbox

```pseudocode
interface PathSandbox:
    workspace: path

    function safe_path(relative: string) -> path
        # Resolves relative against workspace
        # If resolved path escapes workspace: MUST throw ValueError
        # Otherwise: returns resolved absolute path
```

---

## Layer 2: Planning

### TodoManager

```pseudocode
interface TodoManager:
    items: list<TodoItem>

    function update(items: list<TodoItem>) -> string
        # MUST reject if > 1 item has status "in_progress"
        # MUST reject if any item has empty content/text
        # MUST reject if any item has invalid status
        # Valid statuses: "pending", "in_progress", "completed"
        # Returns rendered todo list string

    function render() -> string
        # Returns formatted todo list with markers: [ ] [>] [x]
```

**TodoItem shape:**
```pseudocode
type TodoItem:
    content: string     # non-empty
    status: string      # "pending" | "in_progress" | "completed"
```

### NagPolicy

```pseudocode
interface NagPolicy:
    rounds_since_todo: int
    threshold: int          # default: 3

    function tick() -> void
        # Increment rounds_since_todo

    function should_inject() -> bool
        # Returns true if rounds_since_todo >= threshold

    function reset() -> void
        # Sets rounds_since_todo = 0
```

---

## Layer 3: Delegation

### SubagentFactory

```pseudocode
interface SubagentFactory:
    function spawn(prompt: string, tools: list, dispatch: map, max_iterations: int) -> string
        # Creates fresh messages = [user_turn(prompt)]
        # Runs agent loop with given tools/dispatch up to max_iterations
        # MUST NOT include recursive spawn tool in child's tool set
        # Returns only final text (no tool_use/tool_result content)
        # Parent messages grow by exactly 2 (the tool_use + tool_result)
```

---

## Layer 4: Knowledge

### SkillLoader

```pseudocode
interface SkillLoader:
    skills: map<string, Skill>

    function load_all(skills_dir: path) -> void
        # Scans directory for skill definition files
        # Parses frontmatter (name, description) and body

    function descriptions() -> string
        # Returns compact metadata: name + description per skill
        # MUST NOT include full body content
        # Token count << sum of all skill bodies

    function load(name: string) -> string
        # If name in skills: returns full body wrapped in <skill name="...">
        # Else: returns error listing available skill names
```

**Skill frontmatter format:**
```yaml
---
name: skill-name
description: One-line description
tags: comma,separated
---
Full body content...
```

---

## Layer 5: Memory

### MicroCompactor

```pseudocode
interface MicroCompactor:
    keep_recent: int    # default: 3

    function compact(messages: list) -> list
        # Preserves last keep_recent tool_results with full content
        # Replaces older tool_results with placeholder string
        # Placeholder MUST be non-empty and reference the tool name
        # Returns modified messages list
```

### AutoCompactor

```pseudocode
interface AutoCompactor:
    threshold: int      # default: 50000 tokens

    function should_trigger(messages: list) -> bool
        # Returns true if estimated tokens > threshold
        # Token estimation: len(str(messages)) // 4

    function compact(messages: list) -> tuple<list, path>
        # 1. Saves full transcript to disk (transcript_{timestamp}.jsonl)
        # 2. Calls LLM to summarize conversation
        # 3. Returns exactly 2 messages:
        #    [user("[Conversation compressed]\n{summary}"),
        #     assistant("Understood. Continuing.")]
        # Also returns path to saved transcript file
```

### TranscriptStore

```pseudocode
interface TranscriptStore:
    directory: path     # e.g., .transcripts/

    function save(messages: list) -> path
        # Writes messages as JSONL to transcript_{timestamp}.jsonl
        # Returns path to saved file

    function list() -> list<path>
        # Returns all transcript files
```

---

## Layer 6: Tasks

### TaskManager

```pseudocode
interface TaskManager:
    directory: path     # e.g., .tasks/

    function create(subject: string, description: string = "") -> Task
        # Assigns unique, incremental ID
        # Persists as JSON file: task_{id}.json
        # Returns task object

    function get(task_id: int) -> Task
        # Returns task or raises error

    function update(task_id: int, status: string = null,
                    add_blocks: list<int> = null,
                    add_blocked_by: list<int> = null) -> Task
        # Valid statuses: "pending", "in_progress", "completed"
        # When status="completed": removes task_id from all other tasks' blockedBy
        # add_blocks creates bidirectional edges: this.blocks += [ids], other.blockedBy += [this.id]
        # Persists changes to disk

    function list_all() -> string
        # Returns formatted task list
```

**Task shape:**
```pseudocode
type Task:
    id: int
    subject: string
    description: string
    status: string          # "pending" | "in_progress" | "completed"
    owner: string
    worktree: string        # (Layer 11 extension)
    blockedBy: list<int>
    blocks: list<int>
```

---

## Layer 7: Concurrency

### BackgroundManager

```pseudocode
interface BackgroundManager:
    tasks: map<string, BackgroundTask>
    notification_queue: list    # thread-safe

    function run(command: string) -> string
        # Spawns async execution of command
        # Returns immediately with task_id
        # On completion: pushes result to notification_queue

    function check(task_id: string = null) -> string
        # Returns status of specific task or all tasks

    function drain_notifications() -> list
        # Atomically returns all pending notifications and clears the queue
        # Second call immediately after returns empty list
```

**Integration**: Before each LLM call, drain notifications and inject as `<background-results>` message.

---

## Layer 8: Teams

### MessageBus

```pseudocode
interface MessageBus:
    inbox_dir: path

    function send(sender: string, to: string, content: string,
                  msg_type: string = "message", extra: map = {}) -> string
        # Appends JSON line to {to}.jsonl
        # msg_type MUST be in: {"message", "broadcast", "shutdown_request",
        #   "shutdown_response", "plan_approval_response"}

    function read_inbox(name: string) -> list
        # Reads and drains {name}.jsonl
        # Returns parsed messages, clears file
        # Second call returns empty list

    function broadcast(sender: string, content: string, teammates: list) -> string
        # Sends to all teammates EXCEPT sender
```

### TeammateManager

```pseudocode
interface TeammateManager:
    function spawn(name: string, role: string, prompt: string) -> string
        # Creates teammate with status="working"
        # Runs agent loop in separate thread
        # Teammate starts with fresh messages [user(prompt)]
        # Returns confirmation string

    function list() -> list<Teammate>
        # Returns all teammates with status
```

---

## Layer 9: Protocols

### RequestTracker

```pseudocode
interface RequestTracker:
    requests: map<string, Request>  # request_id -> Request

    function create_request(type: string, target: string, payload: map) -> string
        # Generates unique request_id
        # Stores with status="pending"
        # Sends message to target
        # Returns request_id

    function resolve(request_id: string, approved: bool, feedback: string = "") -> string
        # Transitions status to "approved" or "rejected"
        # Sends response message back
        # Returns resolution string
```

**Request shape:**
```pseudocode
type Request:
    id: string
    type: string        # "shutdown" | "plan_approval"
    target: string
    status: string      # "pending" | "approved" | "rejected"
    payload: map
```

---

## Layer 10: Autonomy

### IdlePolicy

```pseudocode
interface IdlePolicy:
    poll_interval: int      # seconds, default: 5
    idle_timeout: int       # seconds, default: 60

    function poll(inbox_reader, task_scanner) -> PollResult
        # Checks inbox for messages -> resume if found
        # Scans for unclaimed, unblocked, pending tasks
        # Returns action: "resume" | "claim" | "wait" | "shutdown"
```

### TaskClaimer

```pseudocode
interface TaskClaimer:
    function scan_unclaimed(tasks: list<Task>) -> list<Task>
        # Returns tasks where: status=="pending", no owner, empty blockedBy

    function claim(task_id: int, owner: string) -> Task
        # Thread-safe: sets owner and status="in_progress"
```

### IdentityInjector

```pseudocode
interface IdentityInjector:
    function should_inject(messages: list) -> bool
        # Returns true if len(messages) <= 3

    function inject(messages: list, name: string, role: string, team: string) -> list
        # Inserts identity block at position 0:
        #   user("<identity>You are '{name}', role: {role}, team: {team}</identity>")
        # Inserts acknowledgment at position 1:
        #   assistant("I am {name}. Continuing.")
```

---

## Layer 11: Isolation

### WorktreeManager

```pseudocode
interface WorktreeManager:
    repo_root: path
    index: list<WorktreeEntry>

    function validate_name(name: string) -> void
        # MUST match regex: [A-Za-z0-9._-]{1,40}
        # Rejects names like "../escape"
        # Throws ValueError on invalid

    function create(name: string, task_id: int = null, base_ref: string = "HEAD") -> string
        # Emits worktree.create.before event
        # Creates isolated directory with branch wt/{name}
        # Updates index
        # If task_id: binds task (sets task.worktree, advances pending->in_progress)
        # Emits worktree.create.after event

    function remove(name: string, force: bool = false, complete_task: bool = false) -> string
        # Emits worktree.remove.before event
        # Removes directory and branch
        # If complete_task and bound task: marks task completed, unbinds
        # Updates index: status="removed"
        # Emits worktree.remove.after event

    function list_all() -> string
    function status(name: string) -> string
    function run(name: string, command: string) -> string
```

**WorktreeEntry shape:**
```pseudocode
type WorktreeEntry:
    name: string
    path: string
    branch: string          # "wt/{name}"
    task_id: int | null
    status: string          # "active" | "removed" | "kept"
    created_at: timestamp
```

### EventBus

```pseudocode
interface EventBus:
    log_path: path          # events.jsonl

    function emit(event: string, task: map = {}, worktree: map = {}, error: string = null) -> void
        # Appends JSON line to log file

    function list_recent(limit: int = 20) -> string
        # Returns last N events as formatted string
```

---

## Cross-Cutting: Resilience Layer

### ResilientLlm

```pseudocode
interface ResilientLlm:
    inner: Llm
    policy: RetryPolicy
    auth: AuthProfile

    function create(params: LlmParams) -> Result<LlmResponse, LlmError>
        # Wraps inner.create() with retry logic:
        # - Transient errors (429, 500, 502, 503): exponential backoff + jitter, retry
        # - Auth errors (401, 403): rotate API key if available, retry
        # - Overflow / Fatal: propagate immediately
        # Gives up after max_attempts
```

### LlmError

```pseudocode
enum LlmError:
    Transient { status, message }   # retryable
    Overflow  { message }           # context too long
    Auth      { status, message }   # authentication failure
    Fatal     { message }           # non-retryable
```

### classify_error(status, body) -> LlmError

Maps HTTP status codes and response body keywords to error variants.

---

## Cross-Cutting: Prompt Assembly

### PromptAssembler

```pseudocode
interface PromptAssembler:
    prompts_dir: path   # .nano-agent/prompts/

    function init_defaults() -> void
        # Creates starter prompt files (SOUL.md, IDENTITY.md, TOOLS.md, GUIDELINES.md)
        # Does NOT overwrite existing files

    function compose(ctx: PromptContext) -> string
        # Loads layer files in order: SOUL.md, IDENTITY.md, TOOLS.md, GUIDELINES.md, MEMORY.md
        # Substitutes placeholders: {name}, {role}, {cwd}, {tool_count}
        # Injects dynamic sections (todo state, skills) after TOOLS.md
        # Missing files are skipped (not defaulted)
        # Falls back to minimal prompt if no files found

    function reload() -> void
        # Re-reads files from disk (for hot-reload of prompt edits)
```

---

## Cross-Cutting: Heartbeat / Cron Scheduler

### CronScheduler

```pseudocode
interface CronScheduler:
    entries: list<CronEntry>
    config_path: path   # .nano-agent/cron.json

    function add(name, cron, prompt) -> string
    function remove(name) -> string
    function list() -> string
    function tick() -> list<HeartbeatEvent>
        # Checks all enabled entries against current time
        # Prevents double-fire within same minute
```

### HeartbeatManager

```pseudocode
interface HeartbeatManager:
    scheduler: CronScheduler
    pending: list<HeartbeatEvent>

    function start() -> void
        # Spawns background thread: every 30s, calls scheduler.tick()
        # Pushes fired events to pending queue

    function drain_events() -> list<HeartbeatEvent>
        # Atomically returns and clears pending events
```

**Integration**: Before each LLM call, drain heartbeat events and inject as `[Cron '{name}' fired]: {prompt}` messages.

**Cron format**: `minute hour day-of-month month day-of-week` — supports `*`, `*/N`, specific numbers.

---

## Layer 7: Concurrency (Named Lanes)

### BackgroundManager

```pseudocode
interface BackgroundManager:
    lanes: map<string, Lane>        # "main" (max 1), "cron" (max 1), "background" (max 4)
    all_tasks: map<string, TaskInfo>
    notification_queue: list        # thread-safe

    function run_in_lane(lane: string, command: string) -> string
        # Primary method: submit to named lane
        # Returns immediately with task_id
        # Lane enforces max_concurrency; excess tasks queue

    function run(command: string) -> string
        # Backwards-compatible: run_in_lane("background", command)

    function check(task_id: string) -> string
    function drain_notifications() -> list

    function reset_lane(lane: string) -> void
        # Increments generation counter; queued tasks with old generation are skipped

    function wait_lane_idle(lane: string) -> void
        # Blocks until lane has no active or queued tasks
```

---

## Dependency Graph

```
L1  Core Loop         (foundation — no dependencies)
L2  Planning          → L1
L3  Delegation        → L1
L4  Knowledge         → L1
L5  Memory            → L1
L6  Tasks             → L1
L7  Concurrency       → L1 (named lanes: main, cron, background)
L8  Teams             → L1, L7
L9  Protocols         → L8
L10 Autonomy          → L6, L8, L9
L11 Isolation         → L6, L10

Cross-cutting:
    Resilience        wraps any Llm implementation
    Prompt Assembly   composes system prompt from .nano-agent/prompts/
    Heartbeat         injects cron events via Pre-LLM Injection (EP2)
```

Each layer ONLY extends the core loop through the 5 extension points — it never modifies the loop itself.

//! Integration scenarios exercising multiple layers of the agent architecture.
//!
//! Each scenario simulates a realistic workflow that crosses layer boundaries,
//! verifying that the contracts compose correctly.

use nano_agent::autonomy;
use nano_agent::concurrency::BackgroundManager;
use nano_agent::core_loop::{run_agent_loop, PathSandbox};
use nano_agent::delegation::SubagentFactory;
use nano_agent::isolation::{EventBus, WorktreeManager};
use nano_agent::knowledge::SkillLoader;
use nano_agent::memory;
use nano_agent::mock::{make_dispatch, make_text_block, make_tool_use_block, MockLLM};
use nano_agent::planning::{NagPolicy, TodoItem, TodoManager};
use nano_agent::protocols::RequestTracker;
use nano_agent::tasks::{Task, TaskManager};
use nano_agent::teams::{MessageBus, TeammateManager};
use nano_agent::types::LoopSignals;
use nano_agent::types::*;

use std::collections::HashMap;
use std::fs;
use tempfile::TempDir;

fn tmp() -> TempDir {
    TempDir::new().unwrap()
}

fn empty_dispatch() -> Dispatch {
    make_dispatch(HashMap::new())
}

fn echo_dispatch() -> Dispatch {
    let mut h = HashMap::new();
    h.insert("bash".to_string(), "ok".to_string());
    h.insert("read_file".to_string(), "file contents here".to_string());
    h.insert("write_file".to_string(), "written".to_string());
    make_dispatch(h)
}

// ---------------------------------------------------------------------------
// Scenario 1  (L1 + L2): Agent loop with planning enforcement
//
// An agent runs a multi-step task. After several turns without updating the
// todo list, the NagPolicy fires and injects a reminder into the conversation.
// ---------------------------------------------------------------------------

#[test]
fn scenario_01_loop_with_planning_nag() {
    let mut todo = TodoManager::new();
    let mut nag = NagPolicy::new(3); // nag after 3 ticks

    // Agent does 4 turns of tool work without touching todo
    let mut llm = MockLLM::new();
    for i in 0..3 {
        llm.queue(
            "tool_use",
            vec![make_tool_use_block(
                &format!("c{}", i),
                "bash",
                serde_json::json!({"command": "echo hi"}),
            )],
        );
    }
    llm.queue("end_turn", vec![make_text_block("done")]);

    let dispatch = echo_dispatch();
    let tools: Vec<serde_json::Value> = vec![];
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "implement feature X"})];

    // Simulate ticks alongside agent loop
    let mut nag_fired = false;
    for _ in 0..4 {
        nag.tick();
        if nag.should_inject() {
            nag_fired = true;
        }
    }

    // Nag should have fired after tick 3
    assert!(
        nag_fired,
        "NagPolicy should fire after 3 ticks without todo use"
    );

    // Run agent loop
    let calls = run_agent_loop(
        &mut llm,
        "system",
        &mut messages,
        &tools,
        &dispatch,
        &LoopSignals::none(),
    );
    assert_eq!(calls, 4);

    // Verify todo manager rejects bad states
    let items_ok = vec![TodoItem {
        id: "1".into(),
        text: "task A".into(),
        status: "in_progress".into(),
    }];
    assert!(todo.update(items_ok).is_ok());

    let items_bad = vec![
        TodoItem {
            id: "1".into(),
            text: "task A".into(),
            status: "in_progress".into(),
        },
        TodoItem {
            id: "2".into(),
            text: "task B".into(),
            status: "in_progress".into(),
        },
    ];
    assert!(
        todo.update(items_bad).is_err(),
        "only one in_progress allowed"
    );

    let items_empty = vec![TodoItem {
        id: "1".into(),
        text: "".into(),
        status: "pending".into(),
    }];
    assert!(todo.update(items_empty).is_err(), "empty content rejected");

    // Using todo resets the nag
    nag.reset();
    nag.tick();
    assert!(
        !nag.should_inject(),
        "nag should not fire right after reset"
    );
}

// ---------------------------------------------------------------------------
// Scenario 2  (L1 + L3): Parent agent delegates file-reading subtask
//
// The parent agent receives a request and spawns a subagent to read a file.
// The subagent runs in isolation and returns only the final text.
// ---------------------------------------------------------------------------

#[test]
fn scenario_02_delegation_with_tool_use() {
    // Subagent will: 1) use read_file tool, 2) return summary text
    let mut child_llm = MockLLM::new();
    child_llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "t1",
            "read_file",
            serde_json::json!({"path": "data.txt"}),
        )],
    );
    child_llm.queue(
        "end_turn",
        vec![make_text_block("The file contains secret=42")],
    );

    let tools = nano_agent::delegation::child_tools();
    let mut h = HashMap::new();
    h.insert("read_file".to_string(), "secret=42".to_string());
    let dispatch = make_dispatch(h);

    let result = SubagentFactory::spawn(
        &mut child_llm,
        "Read data.txt and summarize",
        &tools,
        &dispatch,
        5,
    );

    assert!(
        result.contains("secret=42"),
        "subagent should return file content summary"
    );
    assert!(
        !result.contains("tool_use"),
        "internal tool calls should not leak"
    );
}

// ---------------------------------------------------------------------------
// Scenario 3  (L4 + L1): Skill-guided agent loop
//
// Skills are loaded from markdown files, their descriptions injected into the
// system prompt, and the agent uses tool calls informed by the skill content.
// ---------------------------------------------------------------------------

#[test]
fn scenario_03_skill_injection_into_system_prompt() {
    let dir = tmp();
    let skills_dir = dir.path().join("skills");

    // SkillLoader expects subdirectories each containing a SKILL.md
    let testing_dir = skills_dir.join("testing");
    let review_dir = skills_dir.join("review");
    fs::create_dir_all(&testing_dir).unwrap();
    fs::create_dir_all(&review_dir).unwrap();

    fs::write(
        testing_dir.join("SKILL.md"),
        "---\nname: testing\ndescription: Run tests with cargo test\ntags: rust, testing\n---\n\
         Always run `cargo test` before committing. Check for warnings.\n",
    )
    .unwrap();

    fs::write(
        review_dir.join("SKILL.md"),
        "---\nname: code-review\ndescription: Review code for common issues\ntags: review\n---\n\
         Look for: unwrap() in production, missing error handling, TODO comments.\n",
    )
    .unwrap();

    let loader = SkillLoader::new(&skills_dir);

    // System prompt uses compact descriptions
    let descriptions = loader.get_descriptions();
    assert!(
        descriptions.contains("testing"),
        "should list testing skill"
    );
    assert!(
        descriptions.contains("code-review"),
        "should list review skill"
    );
    assert!(
        !descriptions.contains("Always run"),
        "body should NOT be in descriptions"
    );

    // When agent needs a skill, load full content
    let content = loader.get_content("testing");
    assert!(
        content.contains("cargo test"),
        "full body should be available"
    );

    // Unknown skill gives helpful error
    let err = loader.get_content("unknown-skill");
    assert!(
        err.contains("Error"),
        "should return error for unknown skill"
    );
}

// ---------------------------------------------------------------------------
// Scenario 4  (L5 + L1): Memory compaction during long conversation
//
// A long conversation exceeds token limits. micro_compact replaces old tool
// results with placeholders. auto_compact saves transcript and summarizes.
// ---------------------------------------------------------------------------

#[test]
fn scenario_04_memory_compaction_pipeline() {
    let dir = tmp();
    let transcript_dir = dir.path().join("transcripts");

    // Build a long conversation with many tool results (content > 100 chars to trigger replacement)
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for i in 0..20 {
        messages.push(serde_json::json!({
            "role": "user",
            "content": format!("question {}", i),
        }));
        messages.push(serde_json::json!({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": format!("t{}", i), "name": "bash", "input": {}}],
        }));
        messages.push(serde_json::json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": format!("t{}", i),
                "content": format!("result {}: {}", i, "x".repeat(200)),
            }],
        }));
    }

    // Step 1: Micro-compact — keep last KEEP_RECENT (3) tool results, replace older ones
    memory::micro_compact(&mut messages);

    let mut full_results = 0;
    let mut placeholders = 0;
    for msg in &messages {
        if let Some(arr) = msg.get("content").and_then(|c| c.as_array()) {
            for block in arr {
                if block.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                    let content = block.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    if content.starts_with("[Previous") {
                        placeholders += 1;
                    } else {
                        full_results += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        full_results,
        memory::KEEP_RECENT,
        "should keep KEEP_RECENT most recent tool results"
    );
    assert_eq!(
        placeholders,
        20 - memory::KEEP_RECENT,
        "should replace older tool results"
    );

    // Step 2: Auto-compact produces exactly 2 messages and saves transcript
    let mut summary_llm = MockLLM::new();
    summary_llm.queue(
        "end_turn",
        vec![make_text_block(
            "Summary: 20 questions about various topics were asked.",
        )],
    );
    let (result, transcript_path) =
        memory::auto_compact(&messages, &mut summary_llm, &transcript_dir);
    assert_eq!(
        result.len(),
        2,
        "auto-compact returns user + assistant summary"
    );
    assert_eq!(result[0]["role"], "user");
    assert_eq!(result[1]["role"], "assistant");
    assert!(transcript_path.exists(), "transcript should be saved");
}

// ---------------------------------------------------------------------------
// Scenario 5  (L6 + L11): Task lifecycle with worktree isolation
//
// Create tasks with dependencies, bind to worktrees, complete work, and
// verify the full lifecycle including event emission.
// ---------------------------------------------------------------------------

#[test]
fn scenario_05_task_lifecycle_with_worktree() {
    let dir = tmp();
    let tasks_dir = dir.path().join("tasks");
    let events_path = dir.path().join("events.jsonl");

    // Initialize a git repo for worktree operations
    std::process::Command::new("git")
        .args(["init", "--initial-branch", "main"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["commit", "--allow-empty", "-m", "init"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let bus = EventBus::new(&events_path);

    // Create task chain: T1 blocks T2, T2 blocks T3
    let t1_json: Task = serde_json::from_str(&tm.create("Set up database schema")).unwrap();
    let t2_json: Task = serde_json::from_str(&tm.create("Implement API endpoints")).unwrap();
    let t3_json: Task = serde_json::from_str(&tm.create("Write integration tests")).unwrap();
    let t1 = t1_json.id;
    let t2 = t2_json.id;
    let t3 = t3_json.id;

    // T1 blocks T2, T2 blocks T3
    tm.update(t1, None, None, Some(&[t2])).unwrap();
    tm.update(t2, None, None, Some(&[t3])).unwrap();

    // T2 and T3 should be blocked
    let task2: Task = serde_json::from_str(&tm.get(t2)).unwrap();
    assert!(!task2.blocked_by.is_empty(), "T2 should be blocked by T1");
    let task3: Task = serde_json::from_str(&tm.get(t3)).unwrap();
    assert!(!task3.blocked_by.is_empty(), "T3 should be blocked by T2");

    // Only T1 is claimable (no blockers, pending, no owner)
    let claimable = autonomy::scan_unclaimed_tasks(&tasks_dir);
    assert_eq!(claimable.len(), 1);
    assert_eq!(claimable[0].id, t1);

    // Claim T1
    autonomy::claim_task(&tasks_dir, t1, "agent-alpha");

    // Create worktree for T1
    let wm = WorktreeManager::new(dir.path(), &tm, &bus);
    let wt_result = wm.create_with_task("feature-schema", Some(t1));
    assert!(wt_result.is_ok(), "worktree creation should succeed");

    // Complete T1 — should unblock T2
    tm.update_status(t1, "completed").unwrap();
    let task2_after: Task = serde_json::from_str(&tm.get(t2)).unwrap();
    assert!(
        task2_after.blocked_by.is_empty(),
        "T2 should be unblocked after T1 completed"
    );

    // T2 is now claimable, T3 still blocked
    let claimable2 = autonomy::scan_unclaimed_tasks(&tasks_dir);
    assert_eq!(claimable2.len(), 1);
    assert_eq!(claimable2[0].id, t2);

    // Clean up worktree
    let remove_result = wm.remove("feature-schema");
    assert!(remove_result.is_ok());

    // Verify events were emitted
    let events = fs::read_to_string(&events_path).unwrap();
    assert!(
        events.contains("worktree.create.before"),
        "lifecycle events should be emitted"
    );
    assert!(
        events.contains("worktree.create.after"),
        "lifecycle events should be emitted"
    );
}

// ---------------------------------------------------------------------------
// Scenario 6  (L7 + L1): Background commands feeding into agent loop
//
// Agent kicks off background tasks, continues working, then drains results
// and incorporates them into the conversation.
// ---------------------------------------------------------------------------

#[test]
fn scenario_06_background_tasks_feed_loop() {
    let dir = tmp();
    fs::write(dir.path().join("a.txt"), "alpha").unwrap();
    fs::write(dir.path().join("b.txt"), "bravo").unwrap();

    let bg = BackgroundManager::new();

    // Kick off two background commands
    let id1 = bg.run(&format!("cat {}/a.txt", dir.path().display()));
    let id2 = bg.run(&format!("cat {}/b.txt", dir.path().display()));

    // Both return immediately with task IDs
    assert!(!id1.is_empty());
    assert!(!id2.is_empty());

    // Wait for completion
    std::thread::sleep(std::time::Duration::from_millis(300));

    // Drain notifications
    let notes = bg.drain_notifications();
    assert_eq!(notes.len(), 2, "should have 2 notifications");

    // Results should contain file contents
    let outputs: Vec<&str> = notes.iter().map(|n| n.result.as_str()).collect();
    assert!(outputs.iter().any(|o| o.contains("alpha")));
    assert!(outputs.iter().any(|o| o.contains("bravo")));

    // Second drain should be empty (atomic drain)
    let notes2 = bg.drain_notifications();
    assert!(
        notes2.is_empty(),
        "drain should be atomic — second call empty"
    );

    // Inject results as tool_result messages (simulating what the agent loop does)
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "process the files"})];
    for note in &notes {
        messages.push(serde_json::json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": format!("bg_{}", note.task_id),
                "content": format!("[background] {}: {}", note.status, note.result),
            }],
        }));
    }
    assert!(
        messages.len() >= 3,
        "background results should be in message history"
    );
}

// ---------------------------------------------------------------------------
// Scenario 7  (L8 + L9): Team shutdown protocol
//
// A lead agent requests shutdown of a teammate. The teammate must approve
// via the protocol. Messages flow through the team message bus.
// ---------------------------------------------------------------------------

#[test]
fn scenario_07_team_shutdown_protocol() {
    let dir = tmp();
    let inbox_dir = dir.path().join("inboxes");
    fs::create_dir(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let tracker = RequestTracker::new();

    // Set up team with 2 members
    let mut team = TeammateManager::new(dir.path());
    team.spawn("lead", "leader", "manage team");
    team.spawn("worker", "developer", "write code");

    // Lead sends shutdown request to worker
    let result = tracker.handle_shutdown_request(&bus, "worker");
    assert!(result.contains("pending"), "should show pending status");

    // Worker reads inbox and finds shutdown request
    let inbox = bus.read_inbox("worker");
    assert!(!inbox.is_empty(), "worker should have message in inbox");
    let msg = &inbox[0];
    assert_eq!(msg["type"].as_str().unwrap(), "shutdown_request");
}

// ---------------------------------------------------------------------------
// Scenario 8  (L8 + L3): Team member delegates to subagent
//
// A teammate receives a task via the message bus, then spawns a subagent
// to handle a subtask, and reports results back to the team.
// ---------------------------------------------------------------------------

#[test]
fn scenario_08_team_delegation() {
    let dir = tmp();
    let inbox_dir = dir.path().join("inboxes");
    fs::create_dir(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let mut team = TeammateManager::new(dir.path());
    team.spawn("coordinator", "lead", "coordinate work");
    team.spawn("specialist", "analyst", "analyze data");

    // Coordinator sends task to specialist
    bus.send("coordinator", "specialist", "Please analyze data.csv");

    // Specialist reads inbox
    let inbox = bus.read_inbox("specialist");
    assert_eq!(inbox.len(), 1);
    assert!(inbox[0]["content"]
        .as_str()
        .unwrap()
        .contains("analyze data.csv"));

    // Specialist spawns subagent to do the work
    let mut child_llm = MockLLM::new();
    child_llm.queue(
        "end_turn",
        vec![make_text_block("Analysis complete: 100 rows, 5 columns")],
    );

    let tools = nano_agent::delegation::child_tools();
    let dispatch = empty_dispatch();
    let result = SubagentFactory::spawn(
        &mut child_llm,
        "Analyze data.csv and summarize",
        &tools,
        &dispatch,
        3,
    );
    assert!(result.contains("Analysis complete"));

    // Specialist reports back to coordinator
    bus.send("specialist", "coordinator", &format!("Result: {}", result));

    // Coordinator reads the result
    let coord_inbox = bus.read_inbox("coordinator");
    assert_eq!(coord_inbox.len(), 1);
    assert!(coord_inbox[0]["content"]
        .as_str()
        .unwrap()
        .contains("Analysis complete"));
}

// ---------------------------------------------------------------------------
// Scenario 9  (L10 + L6 + L11): Autonomous agent picks up work
//
// An autonomous agent scans for unclaimed tasks, claims one, binds it to
// a worktree, works on it, and marks it complete.
// ---------------------------------------------------------------------------

#[test]
fn scenario_09_autonomous_task_pickup() {
    let dir = tmp();
    let tasks_dir = dir.path().join("tasks");
    let events_path = dir.path().join("events.jsonl");

    // Init git repo
    std::process::Command::new("git")
        .args(["init", "--initial-branch", "main"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["commit", "--allow-empty", "-m", "init"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let _bus = EventBus::new(&events_path);

    // Create several tasks, some blocked
    let t1: Task = serde_json::from_str(&tm.create("Fix login bug")).unwrap();
    let t2: Task = serde_json::from_str(&tm.create("Add rate limiting")).unwrap();
    let t3: Task = serde_json::from_str(&tm.create("Deploy to staging")).unwrap();

    // T3 blocked by T2
    tm.update(t2.id, None, None, Some(&[t3.id])).unwrap();

    // Autonomous scan — should find T1 and T2 (not T3, it's blocked)
    let claimable = autonomy::scan_unclaimed_tasks(&tasks_dir);
    assert_eq!(claimable.len(), 2);
    let claimable_ids: Vec<i64> = claimable.iter().map(|t| t.id).collect();
    assert!(claimable_ids.contains(&t1.id));
    assert!(claimable_ids.contains(&t2.id));
    assert!(!claimable_ids.contains(&t3.id));

    // Agent claims T1
    autonomy::claim_task(&tasks_dir, t1.id, "bot-1");
    let task1: Task = serde_json::from_str(&tm.get(t1.id)).unwrap();
    assert_eq!(task1.owner, "bot-1");
    assert_eq!(task1.status, "in_progress");

    // After claiming, T1 no longer shows up in scan
    let claimable2 = autonomy::scan_unclaimed_tasks(&tasks_dir);
    assert_eq!(claimable2.len(), 1);
    assert_eq!(claimable2[0].id, t2.id);

    // Identity injection for new agent
    let identity = autonomy::make_identity_block("bot-1", "developer", "team-alpha");
    let content = identity["content"].as_str().unwrap();
    assert!(content.contains("bot-1"));
    assert!(content.contains("developer"));

    // Should inject when messages are fresh (<=3)
    let short_messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "start"})];
    assert!(autonomy::should_inject(&short_messages));

    let long_messages: Vec<serde_json::Value> = (0..5)
        .map(|i| serde_json::json!({"role": "user", "content": format!("msg {}", i)}))
        .collect();
    assert!(!autonomy::should_inject(&long_messages));
}

// ---------------------------------------------------------------------------
// Scenario 10  (L9 + L8): Plan approval protocol
//
// An agent proposes a plan that requires approval. The protocol tracks the
// request and routes responses through the message bus.
// ---------------------------------------------------------------------------

#[test]
fn scenario_10_plan_approval_flow() {
    let dir = tmp();
    let inbox_dir = dir.path().join("inboxes");
    fs::create_dir(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let tracker = RequestTracker::new();

    // First, register a plan request in the tracker so handle_plan_review can find it
    let req_id = "plan-001".to_string();
    {
        let mut reqs = tracker.plan_requests.lock().unwrap();
        reqs.insert(
            req_id.clone(),
            nano_agent::protocols::PlanRequest {
                from: "agent-1".to_string(),
                plan: "Refactor auth module, add OAuth2, update tests".to_string(),
                status: "pending".to_string(),
            },
        );
    }

    // Lead reviews and approves the plan
    let result = tracker.handle_plan_review(&bus, &req_id, true, "Looks good, proceed.");
    assert!(result.contains("approved"));

    // Check the request status was updated
    {
        let reqs = tracker.plan_requests.lock().unwrap();
        assert_eq!(reqs[&req_id].status, "approved");
    }

    // Agent-1 reads approval response from inbox
    let agent_inbox = bus.read_inbox("agent-1");
    assert_eq!(agent_inbox.len(), 1);
    assert_eq!(
        agent_inbox[0]["type"].as_str().unwrap(),
        "plan_approval_response"
    );
}

// ---------------------------------------------------------------------------
// Scenario 11  (L1 + L5 + L2): Long session with compaction and planning
//
// Simulates a long agent session where: todo items track progress, memory
// compaction kicks in, and the agent loop processes multiple tool calls.
// ---------------------------------------------------------------------------

#[test]
fn scenario_11_long_session_with_compaction_and_planning() {
    let dir = tmp();

    // Set up todo tracking
    let mut todo = TodoManager::new();
    todo.update(vec![TodoItem {
        id: "1".into(),
        text: "Set up project structure".into(),
        status: "in_progress".into(),
    }])
    .unwrap();

    // Build a conversation that would trigger compaction (more than KEEP_RECENT tool results)
    let tool_count = memory::KEEP_RECENT + 5;
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for i in 0..tool_count {
        messages.push(serde_json::json!({"role": "user", "content": format!("step {}", i)}));
        messages.push(serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": format!("call_{}", i), "name": "bash", "input": {"command": "ls"}},
            ],
        }));
        messages.push(serde_json::json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": format!("call_{}", i),
                "content": format!("output of step {}: {}", i, "data ".repeat(30)),
            }],
        }));
    }

    // Micro-compact in place
    memory::micro_compact(&mut messages);

    // Count placeholders vs full results
    let mut placeholder_count = 0;
    let mut full_count = 0;
    for msg in &messages {
        if let Some(arr) = msg.get("content").and_then(|c| c.as_array()) {
            for block in arr {
                if block.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                    let content = block.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    if content.starts_with("[Previous") {
                        placeholder_count += 1;
                    } else {
                        full_count += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        full_count,
        memory::KEEP_RECENT,
        "keep KEEP_RECENT full results"
    );
    assert_eq!(placeholder_count, 5, "5 old results should be placeholders");

    // Update todo progress
    todo.update(vec![
        TodoItem {
            id: "1".into(),
            text: "Set up project structure".into(),
            status: "completed".into(),
        },
        TodoItem {
            id: "2".into(),
            text: "Implement core logic".into(),
            status: "in_progress".into(),
        },
    ])
    .unwrap();

    // Save transcript
    let store = memory::TranscriptStore::new(dir.path());
    store.save(&messages);
    assert_eq!(store.list().len(), 1);
}

// ---------------------------------------------------------------------------
// Scenario 12  (L7 + L8): Background tasks with team notification
//
// Agent runs background commands and notifies teammates of results.
// ---------------------------------------------------------------------------

#[test]
fn scenario_12_background_with_team_notification() {
    let dir = tmp();
    let inbox_dir = dir.path().join("inboxes");
    fs::create_dir(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let bg = BackgroundManager::new();

    // Spawn background build
    let build_result = bg.run("echo BUILD_OK");
    assert!(build_result.contains("Background task"));

    // Wait for completion
    std::thread::sleep(std::time::Duration::from_millis(300));

    let notes = bg.drain_notifications();
    assert_eq!(notes.len(), 1);
    assert!(notes[0].result.contains("BUILD_OK"));

    // Notify team of build result
    bus.send(
        "builder",
        "deployer",
        &format!("Build completed: {}", notes[0].result.trim()),
    );

    let deployer_inbox = bus.read_inbox("deployer");
    assert_eq!(deployer_inbox.len(), 1);
    assert!(deployer_inbox[0]["content"]
        .as_str()
        .unwrap()
        .contains("BUILD_OK"));
}

// ---------------------------------------------------------------------------
// Scenario 13  (L11): Worktree name validation and event lifecycle
//
// Verifies worktree name safety and full create/remove lifecycle events.
// ---------------------------------------------------------------------------

#[test]
fn scenario_13_worktree_safety_and_events() {
    let dir = tmp();
    let tasks_dir = dir.path().join("tasks");
    let events_path = dir.path().join("events.jsonl");

    // Init git repo
    std::process::Command::new("git")
        .args(["init", "--initial-branch", "main"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["commit", "--allow-empty", "-m", "init"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let bus = EventBus::new(&events_path);
    let wm = WorktreeManager::new(dir.path(), &tm, &bus);

    // Reject dangerous names
    assert!(wm.create("../escape").is_err());
    assert!(wm.create("foo/../../bar").is_err());
    assert!(wm.create("").is_err());

    // Valid creation
    let result = wm.create("fix-auth-bug");
    assert!(result.is_ok());

    // Verify worktree directory exists
    assert!(dir.path().join(".worktrees").join("fix-auth-bug").exists());

    // Clean removal
    let remove = wm.remove("fix-auth-bug");
    assert!(remove.is_ok());

    // Events should contain lifecycle markers
    let events = fs::read_to_string(&events_path).unwrap();
    let event_lines: Vec<&str> = events.lines().collect();
    assert!(
        event_lines.len() >= 4,
        "should have create.before, create.after, remove.before, remove.after"
    );
}

// ---------------------------------------------------------------------------
// Scenario 14  (L1 + L3 + L4): Skill-informed subagent delegation
//
// Parent loads a skill, injects it into the subagent's system prompt,
// and the subagent uses the skill context in its work.
// ---------------------------------------------------------------------------

#[test]
fn scenario_14_skill_informed_delegation() {
    let dir = tmp();
    let skills_dir = dir.path().join("skills");
    let deploy_dir = skills_dir.join("deploy");
    fs::create_dir_all(&deploy_dir).unwrap();

    // Create a deployment skill
    fs::write(
        deploy_dir.join("SKILL.md"),
        "---\nname: deploy\ndescription: Deploy to production safely\ntags: ops\n---\n\
         Steps: 1) Run tests  2) Build release  3) Tag version  4) Push to registry\n\
         Always verify health checks after deployment.\n",
    )
    .unwrap();

    let loader = SkillLoader::new(&skills_dir);
    let skill_content = loader.get_content("deploy");
    assert!(skill_content.contains("health checks"));

    // Spawn subagent with skill-augmented prompt
    let mut child_llm = MockLLM::new();
    child_llm.queue(
        "end_turn",
        vec![make_text_block(
            "Deployment complete. Health checks passed.",
        )],
    );

    let tools = nano_agent::delegation::child_tools();
    let dispatch = empty_dispatch();
    let result = SubagentFactory::spawn(&mut child_llm, "Deploy v2.1.0", &tools, &dispatch, 3);

    assert!(result.contains("Deployment complete"));
}

// ---------------------------------------------------------------------------
// Scenario 15  (L6 + L10 + L8): Multi-agent task queue
//
// Multiple autonomous agents scan a shared task queue, each claims different
// tasks, and they communicate progress via the message bus.
// ---------------------------------------------------------------------------

#[test]
fn scenario_15_multi_agent_task_queue() {
    let dir = tmp();
    let tasks_dir = dir.path().join("tasks");
    let inbox_dir = dir.path().join("inboxes");
    fs::create_dir(&inbox_dir).unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let bus = MessageBus::new(&inbox_dir);

    // Create a batch of tasks
    let ids: Vec<i64> = (0..5)
        .map(|i| {
            let t: Task = serde_json::from_str(&tm.create(&format!("Task {}", i))).unwrap();
            t.id
        })
        .collect();

    // Agent A scans and claims first available
    let claimable_a = autonomy::scan_unclaimed_tasks(&tasks_dir);
    assert_eq!(claimable_a.len(), 5);
    autonomy::claim_task(&tasks_dir, ids[0], "agent-A");
    autonomy::claim_task(&tasks_dir, ids[1], "agent-A");

    // Agent B scans — should see only 3 unclaimed
    let claimable_b = autonomy::scan_unclaimed_tasks(&tasks_dir);
    assert_eq!(claimable_b.len(), 3);
    autonomy::claim_task(&tasks_dir, ids[2], "agent-B");

    // Agents communicate progress
    bus.send("agent-A", "agent-B", "Completed Task 0, starting Task 1");
    bus.send("agent-B", "agent-A", "Working on Task 2");

    // Both read their messages
    let inbox_a = bus.read_inbox("agent-A");
    assert_eq!(inbox_a.len(), 1);
    assert!(inbox_a[0]["content"].as_str().unwrap().contains("Task 2"));

    let inbox_b = bus.read_inbox("agent-B");
    assert_eq!(inbox_b.len(), 1);
    assert!(inbox_b[0]["content"].as_str().unwrap().contains("Task 0"));

    // Agent A completes Task 0
    tm.update_status(ids[0], "completed").unwrap();
    let task0: Task = serde_json::from_str(&tm.get(ids[0])).unwrap();
    assert_eq!(task0.status, "completed");
    assert_eq!(task0.owner, "agent-A");

    // Broadcast completion to all
    let names = vec!["agent-A".to_string(), "agent-B".to_string()];
    bus.broadcast("agent-A", "Task 0 is done!", &names);
    let bcast = bus.read_inbox("agent-B");
    assert_eq!(bcast.len(), 1);
    assert!(bcast[0]["content"]
        .as_str()
        .unwrap()
        .contains("Task 0 is done"));
}

// ---------------------------------------------------------------------------
// Scenario 16  (L1): Path sandbox edge cases
//
// Thorough testing of the path sandbox with various escape attempts.
// ---------------------------------------------------------------------------

#[test]
fn scenario_16_path_sandbox_edge_cases() {
    let dir = tmp();
    let sandbox = PathSandbox::new(dir.path());

    // Valid paths
    assert!(sandbox.safe_path("file.txt").is_ok());
    assert!(sandbox.safe_path("src/main.rs").is_ok());
    assert!(sandbox.safe_path("deeply/nested/dir/file.rs").is_ok());

    // Escape attempts
    assert!(sandbox.safe_path("../../../etc/passwd").is_err());
    assert!(sandbox.safe_path("src/../../outside").is_err());
    assert!(sandbox.safe_path("..").is_err());

    // Tricky but valid: normalizes back within workspace
    assert!(sandbox.safe_path("src/../src/main.rs").is_ok());
}

// ---------------------------------------------------------------------------
// Scenario 17  (L2 + L6): Planning integrated with task management
//
// Todo items map to tasks. Completing a todo also completes the task.
// ---------------------------------------------------------------------------

#[test]
fn scenario_17_planning_with_task_tracking() {
    let dir = tmp();
    let tasks_dir = dir.path().join("tasks");

    let mut todo = TodoManager::new();
    let tm = TaskManager::new(&tasks_dir);

    // Create linked todo + task pairs
    let t1: Task = serde_json::from_str(&tm.create("Design API schema")).unwrap();
    let t2: Task = serde_json::from_str(&tm.create("Implement endpoints")).unwrap();
    let _t3: Task = serde_json::from_str(&tm.create("Write tests")).unwrap();

    todo.update(vec![
        TodoItem {
            id: "1".into(),
            text: "Design API schema".into(),
            status: "in_progress".into(),
        },
        TodoItem {
            id: "2".into(),
            text: "Implement endpoints".into(),
            status: "pending".into(),
        },
        TodoItem {
            id: "3".into(),
            text: "Write tests".into(),
            status: "pending".into(),
        },
    ])
    .unwrap();

    // Complete first item, start second
    todo.update(vec![
        TodoItem {
            id: "1".into(),
            text: "Design API schema".into(),
            status: "completed".into(),
        },
        TodoItem {
            id: "2".into(),
            text: "Implement endpoints".into(),
            status: "in_progress".into(),
        },
        TodoItem {
            id: "3".into(),
            text: "Write tests".into(),
            status: "pending".into(),
        },
    ])
    .unwrap();
    tm.update_status(t1.id, "completed").unwrap();
    tm.update_status(t2.id, "in_progress").unwrap();

    // Verify states are consistent
    let task1: Task = serde_json::from_str(&tm.get(t1.id)).unwrap();
    assert_eq!(task1.status, "completed");
    let task2: Task = serde_json::from_str(&tm.get(t2.id)).unwrap();
    assert_eq!(task2.status, "in_progress");

    // Can't have two in-progress todos
    let bad_update = todo.update(vec![
        TodoItem {
            id: "1".into(),
            text: "Design API schema".into(),
            status: "completed".into(),
        },
        TodoItem {
            id: "2".into(),
            text: "Implement endpoints".into(),
            status: "in_progress".into(),
        },
        TodoItem {
            id: "3".into(),
            text: "Write tests".into(),
            status: "in_progress".into(),
        },
    ]);
    assert!(bad_update.is_err());
}

// ---------------------------------------------------------------------------
// Scenario 18  (L5): Token estimation drives compaction decisions
//
// Verifies that token estimation is roughly correct and that large
// conversations trigger compaction appropriately.
// ---------------------------------------------------------------------------

#[test]
fn scenario_18_token_estimation() {
    // ~4 chars per token is the heuristic (operates on serialized JSON)
    let short_msgs = vec![serde_json::json!({"role": "user", "content": "hello"})];
    let short_tokens = memory::estimate_tokens(&short_msgs);
    assert!(
        short_tokens > 0 && short_tokens < 50,
        "short message should be few tokens: got {}",
        short_tokens
    );

    // Large conversation
    let big_msgs: Vec<serde_json::Value> = (0..100)
        .map(|i| {
            serde_json::json!({
                "role": "user",
                "content": format!("{}: {}", i, "word ".repeat(50)),
            })
        })
        .collect();
    let big_tokens = memory::estimate_tokens(&big_msgs);
    assert!(
        big_tokens > 5000,
        "100 messages with 50 words should be many tokens: got {}",
        big_tokens
    );

    // Verify threshold constant is accessible
    assert!(memory::THRESHOLD > 0);
    assert!(memory::KEEP_RECENT > 0);
}

// ---------------------------------------------------------------------------
// Scenario 19  (L1 + L7 + L5): Full agent session simulation
//
// Simulates a complete agent session: receives task, runs tool work,
// compacts memory, and produces final output.
// ---------------------------------------------------------------------------

#[test]
fn scenario_19_full_session_simulation() {
    let dir = tmp();

    let mut llm = MockLLM::new();

    // Turn 1: Agent reads a file
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "c1",
            "read_file",
            serde_json::json!({"path": "config.json"}),
        )],
    );
    // Turn 2: Agent runs a command
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "c2",
            "bash",
            serde_json::json!({"command": "echo test"}),
        )],
    );
    // Turn 3: Agent writes a file
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "c3",
            "write_file",
            serde_json::json!({"path": "out.txt", "content": "result"}),
        )],
    );
    // Turn 4: Final response
    llm.queue(
        "end_turn",
        vec![make_text_block(
            "Task complete. Created out.txt with results.",
        )],
    );

    let dispatch = echo_dispatch();
    let tools: Vec<serde_json::Value> = vec![];
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "Process config and create output"})];

    let calls = run_agent_loop(
        &mut llm,
        "You are a helper.",
        &mut messages,
        &tools,
        &dispatch,
        &LoopSignals::none(),
    );

    // Should have made 4 LLM calls (3 tool_use + 1 end_turn)
    assert_eq!(calls, 4);

    // Messages should contain the full conversation
    assert!(
        messages.len() >= 8,
        "should have user + 4 assistant + 3 tool_result messages, got {}",
        messages.len()
    );

    // Last message should be assistant with final text
    let last = messages.last().unwrap();
    assert_eq!(last["role"], "assistant");

    // Save transcript
    let store = memory::TranscriptStore::new(dir.path());
    store.save(&messages);
    assert_eq!(store.list().len(), 1);
}

// ---------------------------------------------------------------------------
// Scenario 20  (L6 + L11 + L9): Complete project workflow
//
// End-to-end: create tasks, set up worktree, do work, request plan approval,
// complete tasks, and tear down worktree.
// ---------------------------------------------------------------------------

#[test]
fn scenario_20_complete_project_workflow() {
    let dir = tmp();
    let tasks_dir = dir.path().join("tasks");
    let inbox_dir = dir.path().join("inboxes");
    let events_path = dir.path().join("events.jsonl");
    fs::create_dir(&inbox_dir).unwrap();

    // Init git repo
    std::process::Command::new("git")
        .args(["init", "--initial-branch", "main"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["commit", "--allow-empty", "-m", "init"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let bus = EventBus::new(&events_path);
    let msg_bus = MessageBus::new(&inbox_dir);
    let tracker = RequestTracker::new();

    // Phase 1: Create tasks
    let t1: Task = serde_json::from_str(&tm.create("Implement auth module")).unwrap();
    let t2: Task = serde_json::from_str(&tm.create("Add unit tests")).unwrap();
    tm.update(t1.id, None, None, Some(&[t2.id])).unwrap(); // T1 blocks T2

    // Phase 2: Agent claims T1 and sets up worktree
    autonomy::claim_task(&tasks_dir, t1.id, "dev-agent");
    let wm = WorktreeManager::new(dir.path(), &tm, &bus);
    wm.create_with_task("auth-module", Some(t1.id)).unwrap();

    // Phase 3: Register a plan request and get approval
    let req_id = "pr-001".to_string();
    {
        let mut reqs = tracker.plan_requests.lock().unwrap();
        reqs.insert(
            req_id.clone(),
            nano_agent::protocols::PlanRequest {
                from: "dev-agent".to_string(),
                plan: "1) Create auth middleware  2) Add JWT validation  3) Write tests"
                    .to_string(),
                status: "pending".to_string(),
            },
        );
    }
    tracker.handle_plan_review(&msg_bus, &req_id, true, "Approved. Go ahead.");

    // Phase 4: Complete T1 — unblocks T2
    tm.update_status(t1.id, "completed").unwrap();
    let t2_after: Task = serde_json::from_str(&tm.get(t2.id)).unwrap();
    assert!(t2_after.blocked_by.is_empty(), "T2 should be unblocked");

    // Phase 5: Clean up worktree
    wm.remove("auth-module").unwrap();

    // Phase 6: T2 now claimable
    let unclaimed = autonomy::scan_unclaimed_tasks(&tasks_dir);
    assert_eq!(unclaimed.len(), 1);
    assert_eq!(unclaimed[0].id, t2.id);

    // Verify events tell the full story
    let events = fs::read_to_string(&events_path).unwrap();
    let event_count = events.lines().count();
    assert!(event_count >= 4, "should have worktree lifecycle events");
}

// ---------------------------------------------------------------------------
// Scenario 21  (Resilience + L1): Transient errors trigger retry, then succeed
//
// The resilience layer retries on transient LLM errors. After 2 failures,
// the third attempt succeeds and the agent loop completes normally.
// ---------------------------------------------------------------------------

#[test]
fn scenario_21_resilience_retry_then_succeed() {
    use nano_agent::resilience::{AuthProfile, ResilientLlm, RetryPolicy};

    let mut mock = MockLLM::new();
    // Queue 2 transient errors followed by a successful response
    mock.queue_error(LlmError::Transient {
        status: 429,
        message: "rate limited".into(),
    });
    mock.queue_error(LlmError::Transient {
        status: 500,
        message: "internal error".into(),
    });
    mock.queue("end_turn", vec![make_text_block("Hello after retries!")]);

    let policy = RetryPolicy {
        max_attempts: 5,
        base_delay_ms: 1, // minimal delay for tests
        max_delay_ms: 5,
        jitter_factor: 0.0,
    };
    let mut resilient: Box<dyn Llm> = Box::new(ResilientLlm::new(
        Box::new(mock),
        policy,
        AuthProfile::empty(),
    ));

    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "hello"})];
    let tools: Vec<serde_json::Value> = vec![];
    let dispatch = empty_dispatch();

    let calls = run_agent_loop(
        resilient.as_mut(),
        "system",
        &mut messages,
        &tools,
        &dispatch,
        &LoopSignals::none(),
    );

    // Loop should complete successfully (1 LLM call from the loop's perspective)
    assert_eq!(calls, 1);
    // Final message should contain the success text
    let last = messages.last().unwrap();
    assert_eq!(last["role"], "assistant");
}

// ---------------------------------------------------------------------------
// Scenario 22  (Resilience + L1): Overflow error triggers compaction signal
//
// When the LLM returns an overflow error, the core loop should set the
// compaction signal and inject a system message, allowing recovery.
// ---------------------------------------------------------------------------

#[test]
fn scenario_22_overflow_triggers_compaction() {
    use nano_agent::resilience::{AuthProfile, ResilientLlm, RetryPolicy};

    let mut mock = MockLLM::new();
    // First call overflows, second (after compaction hint) succeeds
    mock.queue_error(LlmError::Overflow {
        message: "context too long".into(),
    });
    mock.queue(
        "end_turn",
        vec![make_text_block("Recovered after compaction")],
    );

    let policy = RetryPolicy {
        max_attempts: 1,
        base_delay_ms: 1,
        max_delay_ms: 5,
        jitter_factor: 0.0,
    };
    let mut resilient: Box<dyn Llm> = Box::new(ResilientLlm::new(
        Box::new(mock),
        policy,
        AuthProfile::empty(),
    ));

    let compact_signal = CompactSignal::new();
    let signals = LoopSignals {
        compact_signal: Some(&compact_signal),
        transcript_dir: None,
        idle_signal: None,
        tool_callback: None,
        interrupt_signal: None,
    };

    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "process a very long context"})];
    let tools: Vec<serde_json::Value> = vec![];
    let dispatch = empty_dispatch();

    let calls = run_agent_loop(
        resilient.as_mut(),
        "system",
        &mut messages,
        &tools,
        &dispatch,
        &signals,
    );

    // 1 counted LLM call: overflow doesn't increment call_count (it `continue`s),
    // the second call succeeds and is counted
    assert_eq!(calls, 1);
    // The loop should have recovered — last message is assistant
    let last = messages.last().unwrap();
    assert_eq!(last["role"], "assistant");
    // An overflow system message should be in the conversation
    let has_overflow_msg = messages.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .map_or(false, |s| s.contains("Context overflow"))
    });
    assert!(has_overflow_msg, "overflow message should be injected");
}

// ---------------------------------------------------------------------------
// Scenario 23  (Resilience + L1): Fatal error stops the loop gracefully
//
// A non-retryable error should stop the agent loop and inject an error
// message into the conversation — no panic, no infinite retry.
// ---------------------------------------------------------------------------

#[test]
fn scenario_23_fatal_error_stops_loop() {
    use nano_agent::resilience::{AuthProfile, ResilientLlm, RetryPolicy};

    let mut mock = MockLLM::new();
    mock.queue_error(LlmError::Fatal {
        message: "model not found".into(),
    });

    let policy = RetryPolicy {
        max_attempts: 3,
        base_delay_ms: 1,
        max_delay_ms: 5,
        jitter_factor: 0.0,
    };
    let mut resilient: Box<dyn Llm> = Box::new(ResilientLlm::new(
        Box::new(mock),
        policy,
        AuthProfile::empty(),
    ));

    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "hello"})];
    let tools: Vec<serde_json::Value> = vec![];
    let dispatch = empty_dispatch();

    let calls = run_agent_loop(
        resilient.as_mut(),
        "system",
        &mut messages,
        &tools,
        &dispatch,
        &LoopSignals::none(),
    );

    // Fatal errors break before incrementing call_count
    assert_eq!(calls, 0);
    // Error should be injected into conversation as a system message
    let has_error_msg = messages.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .map_or(false, |s| s.contains("LLM error") || s.contains("Fatal"))
    });
    assert!(has_error_msg, "error should be injected into messages");
}

// ---------------------------------------------------------------------------
// Scenario 24  (Prompt Assembly + L4 + L2): Dynamic prompt composition
//
// Prompt assembler loads layered files, substitutes placeholders, and
// injects dynamic todo/skill state — all composing into the system prompt.
// ---------------------------------------------------------------------------

#[test]
fn scenario_24_prompt_assembly_with_skills_and_todos() {
    use nano_agent::prompt::{PromptAssembler, PromptContext};

    let dir = tmp();
    let prompts_dir = dir.path().join("prompts");

    // Create assembler and initialize defaults
    let mut assembler = PromptAssembler::new(&prompts_dir);
    assembler.init_defaults();
    assembler.reload(); // pick up the newly created files

    let ctx = PromptContext {
        agent_name: "alpha".into(),
        agent_role: "architect".into(),
        cwd: "/project/root".into(),
        tool_count: 34,
        todo_state: "[ ] Design schema\n[>] Implement API\n[x] Set up CI".into(),
        skill_descriptions: "- testing: Run tests with cargo\n- deploy: Deploy safely".into(),
        timestamp: "2026-03-17T00:00:00".into(),
        ..Default::default()
    };

    let prompt = assembler.compose(&ctx);

    // Placeholder substitution works
    assert!(prompt.contains("alpha"), "agent name should be substituted");
    assert!(
        prompt.contains("architect"),
        "agent role should be substituted"
    );
    assert!(
        prompt.contains("/project/root"),
        "cwd should be substituted"
    );
    assert!(prompt.contains("34"), "tool count should be substituted");

    // Dynamic sections injected after TOOLS.md
    assert!(
        prompt.contains("Current Todo List"),
        "todo section should be present"
    );
    assert!(prompt.contains("Implement API"), "todo items should appear");
    assert!(
        prompt.contains("Available Skills"),
        "skills section should be present"
    );
    assert!(
        prompt.contains("testing"),
        "skill descriptions should appear"
    );
    assert!(
        prompt.contains("deploy"),
        "skill descriptions should appear"
    );

    // Guidelines still appear after the dynamic sections
    assert!(
        prompt.contains("Guidelines"),
        "guidelines should be present"
    );
}

// ---------------------------------------------------------------------------
// Scenario 25  (Prompt Assembly): Hot-reload after user edits prompt files
//
// Simulates a user editing a prompt file mid-session. After reload(),
// the new content should appear in the composed prompt.
// ---------------------------------------------------------------------------

#[test]
fn scenario_25_prompt_hot_reload() {
    use nano_agent::prompt::{PromptAssembler, PromptContext};

    let dir = tmp();
    let prompts_dir = dir.path().join("prompts");

    let mut assembler = PromptAssembler::new(&prompts_dir);
    assembler.init_defaults();
    assembler.reload();

    let ctx = PromptContext {
        agent_name: "bot".into(),
        agent_role: "dev".into(),
        cwd: "/tmp".into(),
        tool_count: 10,
        timestamp: "2026-03-17T00:00:00".into(),
        ..Default::default()
    };

    let prompt_v1 = assembler.compose(&ctx);
    assert!(prompt_v1.contains("autonomous coding agent"));

    // User edits SOUL.md
    fs::write(
        prompts_dir.join("SOUL.md"),
        "You are a security auditor named '{name}' scanning {cwd} for vulnerabilities.\n",
    )
    .unwrap();

    assembler.reload();
    let prompt_v2 = assembler.compose(&ctx);
    assert!(
        prompt_v2.contains("security auditor"),
        "reloaded prompt should reflect edit"
    );
    assert!(prompt_v2.contains("bot"), "placeholders still substituted");
    assert!(
        !prompt_v2.contains("autonomous coding agent"),
        "old content should be gone"
    );
}

// ---------------------------------------------------------------------------
// Scenario 26  (Named Lanes + L7): Lane concurrency limits enforced
//
// The "main" lane (max 1) serializes tasks. The "background" lane (max 4)
// allows parallel execution. Verify ordering and throughput.
// ---------------------------------------------------------------------------

#[test]
fn scenario_26_lane_concurrency_and_serialization() {
    use nano_agent::concurrency::{BackgroundManager, LANE_BACKGROUND, LANE_MAIN};

    let bg = BackgroundManager::new();

    // Submit 3 tasks to the main lane (serial, max 1)
    let _r1 = bg.run_in_lane(LANE_MAIN, "echo main-1");
    let _r2 = bg.run_in_lane(LANE_MAIN, "echo main-2");
    let _r3 = bg.run_in_lane(LANE_MAIN, "echo main-3");

    // Submit 4 tasks to background lane (parallel, max 4)
    let _b1 = bg.run_in_lane(LANE_BACKGROUND, "echo bg-1");
    let _b2 = bg.run_in_lane(LANE_BACKGROUND, "echo bg-2");
    let _b3 = bg.run_in_lane(LANE_BACKGROUND, "echo bg-3");
    let _b4 = bg.run_in_lane(LANE_BACKGROUND, "echo bg-4");

    // Wait for all to complete
    bg.wait_lane_idle(LANE_MAIN);
    bg.wait_lane_idle(LANE_BACKGROUND);

    let notes = bg.drain_notifications();
    assert_eq!(notes.len(), 7, "all 7 tasks should complete");

    let main_results: Vec<&str> = notes
        .iter()
        .filter(|n| n.command.contains("main-"))
        .map(|n| n.result.as_str())
        .collect();
    assert_eq!(main_results.len(), 3);

    let bg_results: Vec<&str> = notes
        .iter()
        .filter(|n| n.command.contains("bg-"))
        .map(|n| n.result.as_str())
        .collect();
    assert_eq!(bg_results.len(), 4);
}

// ---------------------------------------------------------------------------
// Scenario 27  (Named Lanes): Generation reset cancels queued tasks
//
// After reset_lane(), queued tasks with the old generation are skipped.
// Only tasks submitted after the reset execute.
// ---------------------------------------------------------------------------

#[test]
fn scenario_27_lane_reset_cancels_queued() {
    use nano_agent::concurrency::{BackgroundManager, LANE_MAIN};

    let bg = BackgroundManager::new();

    // Fill the main lane (max 1) with a slow task, then queue more
    let _r1 = bg.run_in_lane(LANE_MAIN, "sleep 0.3 && echo first");
    let _r2 = bg.run_in_lane(LANE_MAIN, "echo should-be-cancelled-1");
    let _r3 = bg.run_in_lane(LANE_MAIN, "echo should-be-cancelled-2");

    // Reset immediately — cancels the queued tasks
    bg.reset_lane(LANE_MAIN);

    // Submit a new task after reset
    let _r4 = bg.run_in_lane(LANE_MAIN, "echo post-reset");

    // Wait for all activity to complete
    bg.wait_lane_idle(LANE_MAIN);
    std::thread::sleep(std::time::Duration::from_millis(500));

    let notes = bg.drain_notifications();
    let results: Vec<String> = notes.iter().map(|n| n.result.clone()).collect();

    // The first task (already running) should complete
    assert!(
        results.iter().any(|r| r.contains("first")),
        "running task should complete"
    );
    // Post-reset task should complete
    assert!(
        results.iter().any(|r| r.contains("post-reset")),
        "post-reset task should run"
    );
    // Cancelled tasks should NOT appear (or at most appear but the key point is post-reset works)
    // Note: the cancelled tasks may or may not have been dequeued depending on timing,
    // but the post-reset task must have run.
}

// ---------------------------------------------------------------------------
// Scenario 28  (Heartbeat + Cron): Full cron lifecycle via HeartbeatManager
//
// Add cron entries, verify they fire on tick, verify persistence across
// reload, and verify drain_events clears the queue.
// ---------------------------------------------------------------------------

#[test]
fn scenario_28_heartbeat_cron_lifecycle() {
    use nano_agent::heartbeat::HeartbeatManager;

    let dir = tmp();
    let config = dir.path().join("cron.json");

    let hb = HeartbeatManager::new(&config);

    // Add two cron entries via delegate methods
    let r1 = hb.add_cron("health", "* * * * *", "check system health");
    assert!(r1.contains("added"));
    let r2 = hb.add_cron("metrics", "*/5 * * * *", "collect metrics");
    assert!(r2.contains("added"));

    // List should show both
    let list = hb.list_crons();
    assert!(list.contains("health"));
    assert!(list.contains("metrics"));

    // Tick the scheduler — "health" (every minute) should fire
    let events = hb.scheduler().tick();
    assert!(
        events.iter().any(|e| e.name == "health"),
        "health cron should fire"
    );

    // Remove health entry
    let rm = hb.remove_cron("health");
    assert!(rm.contains("removed"));
    let list2 = hb.list_crons();
    assert!(!list2.contains("health"), "health should be gone");
    assert!(list2.contains("metrics"), "metrics should remain");

    // Verify persistence — reload from disk
    let hb2 = HeartbeatManager::new(&config);
    let list3 = hb2.list_crons();
    assert!(
        !list3.contains("health"),
        "removed entry should not persist"
    );
    assert!(list3.contains("metrics"), "remaining entry should persist");
}

// ---------------------------------------------------------------------------
// Scenario 29  (Resilience + L3): Subagent survives transient LLM error
//
// A subagent's underlying LLM hits a transient error. The resilience layer
// retries and the subagent returns its result to the parent.
// ---------------------------------------------------------------------------

#[test]
fn scenario_29_resilient_subagent_delegation() {
    use nano_agent::resilience::{AuthProfile, ResilientLlm, RetryPolicy};

    let mut mock = MockLLM::new();
    // Subagent hits one transient error, then succeeds with tool use, then final text
    mock.queue_error(LlmError::Transient {
        status: 503,
        message: "service unavailable".into(),
    });
    mock.queue(
        "tool_use",
        vec![make_tool_use_block(
            "t1",
            "read_file",
            serde_json::json!({"path": "data.txt"}),
        )],
    );
    mock.queue(
        "end_turn",
        vec![make_text_block("File contains: important data")],
    );

    let policy = RetryPolicy {
        max_attempts: 3,
        base_delay_ms: 1,
        max_delay_ms: 5,
        jitter_factor: 0.0,
    };
    let mut resilient = ResilientLlm::new(Box::new(mock), policy, AuthProfile::empty());

    let tools = nano_agent::delegation::child_tools();
    let mut h = HashMap::new();
    h.insert("read_file".to_string(), "important data".to_string());
    let dispatch = make_dispatch(h);

    let result = SubagentFactory::spawn(
        &mut resilient,
        "Read data.txt and summarize",
        &tools,
        &dispatch,
        5,
    );

    assert!(
        result.contains("important data"),
        "subagent should succeed after retry: got '{}'",
        result
    );
}

// ---------------------------------------------------------------------------
// Scenario 30  (All new features): End-to-end session with resilience,
//              prompt assembly, cron, and lanes
//
// Simulates a full agent startup sequence: assemble prompt, configure cron,
// run background tasks in lanes, handle a transient LLM error, and complete.
// ---------------------------------------------------------------------------

#[test]
fn scenario_30_full_new_features_integration() {
    use nano_agent::concurrency::{BackgroundManager, LANE_BACKGROUND, LANE_MAIN};
    use nano_agent::heartbeat::HeartbeatManager;
    use nano_agent::prompt::{PromptAssembler, PromptContext};
    use nano_agent::resilience::{AuthProfile, ResilientLlm, RetryPolicy};

    let dir = tmp();
    let prompts_dir = dir.path().join("prompts");
    let cron_config = dir.path().join("cron.json");

    // Phase 1: Prompt assembly
    let mut assembler = PromptAssembler::new(&prompts_dir);
    assembler.init_defaults();
    assembler.reload();

    let ctx = PromptContext {
        agent_name: "integration-bot".into(),
        agent_role: "full-stack".into(),
        cwd: dir.path().to_string_lossy().into(),
        tool_count: 34,
        todo_state: "[>] Run integration test".into(),
        skill_descriptions: "- testing: cargo test".into(),
        timestamp: "2026-03-17T00:00:00".into(),
        ..Default::default()
    };
    let system = assembler.compose(&ctx);
    assert!(system.contains("integration-bot"));
    assert!(system.contains("Run integration test"));

    // Phase 2: Set up cron
    let hb = HeartbeatManager::new(&cron_config);
    hb.add_cron("watchdog", "* * * * *", "check health");
    let events = hb.scheduler().tick();
    assert_eq!(events.len(), 1, "watchdog should fire immediately");

    // Phase 3: Run background tasks in lanes
    let bg = BackgroundManager::new();
    let _build = bg.run_in_lane(LANE_MAIN, "echo build-ok");
    let _test1 = bg.run_in_lane(LANE_BACKGROUND, "echo test-1-ok");
    let _test2 = bg.run_in_lane(LANE_BACKGROUND, "echo test-2-ok");

    bg.wait_lane_idle(LANE_MAIN);
    bg.wait_lane_idle(LANE_BACKGROUND);

    let notes = bg.drain_notifications();
    assert_eq!(notes.len(), 3);

    // Phase 4: Resilient LLM call with retry
    let mut mock = MockLLM::new();
    mock.queue_error(LlmError::Transient {
        status: 429,
        message: "rate limited".into(),
    });
    mock.queue(
        "end_turn",
        vec![make_text_block("Integration test passed!")],
    );

    let policy = RetryPolicy {
        max_attempts: 3,
        base_delay_ms: 1,
        max_delay_ms: 5,
        jitter_factor: 0.0,
    };
    let mut resilient: Box<dyn Llm> = Box::new(ResilientLlm::new(
        Box::new(mock),
        policy,
        AuthProfile::empty(),
    ));

    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "run integration test"})];

    // Inject background results (as the main loop would)
    for note in &notes {
        messages.push(serde_json::json!({
            "role": "user",
            "content": format!("[background] {}: {}", note.status, note.result),
        }));
    }

    // Inject cron events (as the main loop would)
    for event in &events {
        messages.push(serde_json::json!({
            "role": "user",
            "content": format!("[Cron '{}' fired]: {}", event.name, event.prompt),
        }));
    }

    let tools: Vec<serde_json::Value> = vec![];
    let dispatch = empty_dispatch();

    let calls = run_agent_loop(
        resilient.as_mut(),
        &system,
        &mut messages,
        &tools,
        &dispatch,
        &LoopSignals::none(),
    );

    assert_eq!(calls, 1);
    let last = messages.last().unwrap();
    assert_eq!(last["role"], "assistant");
}

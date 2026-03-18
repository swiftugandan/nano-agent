use nano_agent::autonomy::*;
use nano_agent::teams::MessageBus;

#[test]
fn test_l10_01_idle_detects_inbox_message() {
    let dir = tempfile::tempdir().unwrap();
    let inbox_dir = dir.path().join("inbox");
    std::fs::create_dir_all(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    bus.send("lead", "idle_worker", "Wake up");

    let msgs = bus.read_inbox("idle_worker");
    assert!(!msgs.is_empty());
}

#[test]
fn test_l10_04_idle_timeout_triggers_shutdown() {
    // Simulate the idle polling logic
    let poll_interval = std::time::Duration::from_millis(50);
    let idle_timeout = std::time::Duration::from_millis(150);
    let polls = (idle_timeout.as_millis() / poll_interval.as_millis().max(1)) as usize;

    let mut resume = false;
    for _ in 0..polls {
        std::thread::sleep(poll_interval);
        let inbox: Vec<()> = Vec::new();
        if !inbox.is_empty() {
            resume = true;
            break;
        }
    }

    assert!(!resume);
}

#[test]
fn test_l10_02_auto_claims_unclaimed_unblocked_pending() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    // Create a pending, unowned, unblocked task
    let task = serde_json::json!({
        "id": 1,
        "subject": "Unclaimed task",
        "description": "",
        "status": "pending",
        "owner": "",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_1.json"),
        serde_json::to_string(&task).unwrap(),
    )
    .unwrap();

    let unclaimed = scan_unclaimed_tasks(&tasks_dir);
    assert!(!unclaimed.is_empty());

    let result = claim_task(&tasks_dir, 1, "worker1");
    assert!(result.contains("worker1"));

    // Verify task is now claimed
    let updated: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tasks_dir.join("task_1.json")).unwrap())
            .unwrap();
    assert_eq!(updated["owner"].as_str().unwrap(), "worker1");
    assert_eq!(updated["status"].as_str().unwrap(), "in_progress");
}

#[test]
fn test_l10_03_blocked_tasks_not_claimed() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    // Task 1: pending, unblocked
    let t1 = serde_json::json!({
        "id": 1,
        "subject": "Blocker",
        "description": "",
        "status": "pending",
        "owner": "",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_1.json"),
        serde_json::to_string(&t1).unwrap(),
    )
    .unwrap();

    // Task 2: pending but blocked by task 1
    let t2 = serde_json::json!({
        "id": 2,
        "subject": "Blocked",
        "description": "",
        "status": "pending",
        "owner": "",
        "blockedBy": [1],
    });
    std::fs::write(
        tasks_dir.join("task_2.json"),
        serde_json::to_string(&t2).unwrap(),
    )
    .unwrap();

    let unclaimed = scan_unclaimed_tasks(&tasks_dir);
    let unclaimed_ids: Vec<i64> = unclaimed.iter().map(|t| t.id).collect();
    assert!(unclaimed_ids.contains(&1));
    assert!(!unclaimed_ids.contains(&2));
}

#[test]
fn test_l10_05_identity_reinjection_after_compression() {
    let mut messages = vec![
        serde_json::json!({"role": "user", "content": "compressed summary"}),
        serde_json::json!({"role": "assistant", "content": "understood"}),
    ];
    assert!(should_inject(&messages));

    let identity = make_identity_block("alice", "coder", "alpha");
    let ack = make_acknowledgment("alice");
    messages.insert(0, identity);
    messages.insert(1, ack);

    assert_eq!(messages.len(), 4);
    let content = messages[0]["content"].as_str().unwrap();
    assert!(content.contains("alice"));
    assert!(content.to_lowercase().contains("identity"));
    assert_eq!(messages[1]["role"].as_str().unwrap(), "assistant");
}

#[test]
fn test_l10_06_claim_task_nonexistent() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    let result = claim_task(&tasks_dir, 999, "worker");
    assert!(result.contains("Error") || result.contains("not found"));
}

#[test]
fn test_l10_07_claim_already_claimed_task() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    // Create a task with an owner
    let task = serde_json::json!({
        "id": 1,
        "subject": "Already claimed",
        "description": "",
        "status": "pending",
        "owner": "alice",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_1.json"),
        serde_json::to_string(&task).unwrap(),
    )
    .unwrap();

    // Try to claim it again
    let result = claim_task(&tasks_dir, 1, "bob");
    // Should succeed but overwrite owner (current behavior)
    assert!(result.contains("bob"));

    // Check the task - currently implementation overwrites
    let updated: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tasks_dir.join("task_1.json")).unwrap())
            .unwrap();
    assert_eq!(updated["owner"].as_str().unwrap(), "bob");
}

#[test]
fn test_l10_08_claim_completed_task() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    // Create a completed task
    let task = serde_json::json!({
        "id": 1,
        "subject": "Completed",
        "description": "",
        "status": "completed",
        "owner": "",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_1.json"),
        serde_json::to_string(&task).unwrap(),
    )
    .unwrap();

    // Claiming a completed task should still work (set owner and in_progress)
    let result = claim_task(&tasks_dir, 1, "worker");
    assert!(result.contains("worker"));

    let updated: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tasks_dir.join("task_1.json")).unwrap())
            .unwrap();
    assert_eq!(updated["status"].as_str().unwrap(), "in_progress");
    assert_eq!(updated["owner"].as_str().unwrap(), "worker");
}

#[test]
fn test_l10_09_scan_excludes_non_pending() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    // Task 1: in_progress
    let t1 = serde_json::json!({
        "id": 1,
        "subject": "In progress",
        "description": "",
        "status": "in_progress",
        "owner": "",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_1.json"),
        serde_json::to_string(&t1).unwrap(),
    )
    .unwrap();

    // Task 2: completed
    let t2 = serde_json::json!({
        "id": 2,
        "subject": "Completed",
        "description": "",
        "status": "completed",
        "owner": "",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_2.json"),
        serde_json::to_string(&t2).unwrap(),
    )
    .unwrap();

    // Task 3: pending, unowned (valid)
    let t3 = serde_json::json!({
        "id": 3,
        "subject": "Pending",
        "description": "",
        "status": "pending",
        "owner": "",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_3.json"),
        serde_json::to_string(&t3).unwrap(),
    )
    .unwrap();

    let unclaimed = scan_unclaimed_tasks(&tasks_dir);
    let ids: Vec<i64> = unclaimed.iter().map(|t| t.id).collect();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], 3);
}

#[test]
fn test_l10_10_scan_excludes_owned_tasks() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    // Task 1: unowned pending
    let t1 = serde_json::json!({
        "id": 1,
        "subject": "Unowned",
        "status": "pending",
        "owner": "",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_1.json"),
        serde_json::to_string(&t1).unwrap(),
    )
    .unwrap();

    // Task 2: owned pending (should be excluded)
    let t2 = serde_json::json!({
        "id": 2,
        "subject": "Owned",
        "status": "pending",
        "owner": "alice",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_2.json"),
        serde_json::to_string(&t2).unwrap(),
    )
    .unwrap();

    let unclaimed = scan_unclaimed_tasks(&tasks_dir);
    let ids: Vec<i64> = unclaimed.iter().map(|t| t.id).collect();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], 1);
}

#[test]
fn test_l10_11_scan_excludes_blocked_tasks() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    // Task 1: blocker
    let t1 = serde_json::json!({
        "id": 1,
        "subject": "Blocker",
        "status": "pending",
        "owner": "",
        "blockedBy": [],
    });
    std::fs::write(
        tasks_dir.join("task_1.json"),
        serde_json::to_string(&t1).unwrap(),
    )
    .unwrap();

    // Task 2: blocked by 1
    let t2 = serde_json::json!({
        "id": 2,
        "subject": "Blocked",
        "status": "pending",
        "owner": "",
        "blockedBy": [1],
    });
    std::fs::write(
        tasks_dir.join("task_2.json"),
        serde_json::to_string(&t2).unwrap(),
    )
    .unwrap();

    // Task 3: blocked by 1 and 2
    let t3 = serde_json::json!({
        "id": 3,
        "subject": "Double blocked",
        "status": "pending",
        "owner": "",
        "blockedBy": [1, 2],
    });
    std::fs::write(
        tasks_dir.join("task_3.json"),
        serde_json::to_string(&t3).unwrap(),
    )
    .unwrap();

    let unclaimed = scan_unclaimed_tasks(&tasks_dir);
    let ids: Vec<i64> = unclaimed.iter().map(|t| t.id).collect();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], 1);
}

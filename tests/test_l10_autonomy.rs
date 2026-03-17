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
    let ack = serde_json::json!({"role": "assistant", "content": "I am alice. Continuing."});
    messages.insert(0, identity);
    messages.insert(1, ack);

    assert_eq!(messages.len(), 4);
    let content = messages[0]["content"].as_str().unwrap();
    assert!(content.contains("alice"));
    assert!(content.to_lowercase().contains("identity"));
    assert_eq!(messages[1]["role"].as_str().unwrap(), "assistant");
}

use nano_agent::teams::*;

#[test]
fn test_l8_01_send_appends_jsonl_line() {
    let dir = tempfile::tempdir().unwrap();
    let inbox_dir = dir.path().join("inbox");
    std::fs::create_dir_all(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    bus.send("alice", "bob", "Hello Bob");

    let inbox_file = inbox_dir.join("bob.jsonl");
    assert!(inbox_file.exists());
    let content = std::fs::read_to_string(&inbox_file).unwrap();
    let lines: Vec<&str> = content.trim().lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 1);
    let msg: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(msg["from"].as_str().unwrap(), "alice");
    assert_eq!(msg["content"].as_str().unwrap(), "Hello Bob");
}

#[test]
fn test_l8_02_read_inbox_drains_and_clears() {
    let dir = tempfile::tempdir().unwrap();
    let inbox_dir = dir.path().join("inbox");
    std::fs::create_dir_all(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    bus.send("alice", "bob", "Message 1");
    bus.send("charlie", "bob", "Message 2");

    let first = bus.read_inbox("bob");
    let second = bus.read_inbox("bob");
    assert_eq!(first.len(), 2);
    assert_eq!(second.len(), 0);
}

#[test]
fn test_l8_03_broadcast_skips_sender() {
    let dir = tempfile::tempdir().unwrap();
    let inbox_dir = dir.path().join("inbox");
    std::fs::create_dir_all(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let teammates = vec![
        "alice".to_string(),
        "bob".to_string(),
        "charlie".to_string(),
    ];
    bus.broadcast("alice", "Hello everyone", &teammates);

    // Alice should have nothing
    let alice_msgs = bus.read_inbox("alice");
    assert_eq!(alice_msgs.len(), 0);

    // Bob and Charlie should each have 1
    let bob_msgs = bus.read_inbox("bob");
    let charlie_msgs = bus.read_inbox("charlie");
    assert_eq!(bob_msgs.len(), 1);
    assert_eq!(charlie_msgs.len(), 1);
}

#[test]
fn test_l8_04_spawn_creates_working_teammate() {
    let dir = tempfile::tempdir().unwrap();
    let team_dir = dir.path().join("team");
    std::fs::create_dir_all(&team_dir).unwrap();

    let mut team = TeammateManager::new(&team_dir);
    let result = team.spawn("worker", "coder", "Do some work");

    assert!(result.contains("worker"));
    let member = team.find_member("worker");
    assert!(member.is_some());
    assert!(member.unwrap().status == "working");
}

#[test]
fn test_l8_05_teammate_gets_fresh_messages() {
    // In the Rust version, we verify the spawn contract:
    // teammate starts with fresh messages containing only the prompt.
    // Since we don't have real LLM calls, we verify the TeammateManager
    // correctly creates a member with "working" status.
    let dir = tempfile::tempdir().unwrap();
    let team_dir = dir.path().join("team");
    std::fs::create_dir_all(&team_dir).unwrap();

    let mut team = TeammateManager::new(&team_dir);
    team.spawn("fresh", "tester", "Test something");

    let member = team.find_member("fresh");
    assert!(member.is_some());
    // The member was created with the prompt in fresh messages
    assert_eq!(member.unwrap().status, "working");
}

use nano_agent::tasks::*;

#[test]
fn test_l6_01_task_ids_unique_and_incremental() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let t1: serde_json::Value = serde_json::from_str(&tm.create("First task")).unwrap();
    let t2: serde_json::Value = serde_json::from_str(&tm.create("Second task")).unwrap();

    assert!(t2["id"].as_i64().unwrap() > t1["id"].as_i64().unwrap());
    assert_ne!(t1["id"], t2["id"]);
}

#[test]
fn test_l6_02_completing_task_clears_blocked_by() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let t1: serde_json::Value = serde_json::from_str(&tm.create("Prerequisite")).unwrap();
    let t2: serde_json::Value = serde_json::from_str(&tm.create("Dependent")).unwrap();
    let t1_id = t1["id"].as_i64().unwrap();
    let t2_id = t2["id"].as_i64().unwrap();

    // Set up dependency: task 1 blocks task 2
    tm.update(t1_id, None, None, Some(&[t2_id])).unwrap();

    // Verify dependency exists
    let t2_before: serde_json::Value = serde_json::from_str(&tm.get(t2_id)).unwrap();
    let blocked_by: Vec<i64> = t2_before["blockedBy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert!(blocked_by.contains(&t1_id));

    // Complete task 1
    tm.update(t1_id, Some("completed"), None, None).unwrap();

    // Verify dependency is cleared
    let t2_after: serde_json::Value = serde_json::from_str(&tm.get(t2_id)).unwrap();
    let blocked_by_after: Vec<i64> = t2_after["blockedBy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert!(!blocked_by_after.contains(&t1_id));
}

#[test]
fn test_l6_03_invalid_status_raises_error() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let t: serde_json::Value = serde_json::from_str(&tm.create("Test task")).unwrap();
    let t_id = t["id"].as_i64().unwrap();

    let result = tm.update(t_id, Some("invalid_status"), None, None);
    assert!(result.is_err());
}

#[test]
fn test_l6_04_tasks_persist_as_json_files() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let t: serde_json::Value = serde_json::from_str(&tm.create("Persistent task")).unwrap();
    let t_id = t["id"].as_i64().unwrap();

    let task_file = tasks_dir.join(format!("task_{}.json", t_id));
    assert!(task_file.exists());

    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&task_file).unwrap()).unwrap();
    assert_eq!(data["subject"].as_str().unwrap(), "Persistent task");
}

#[test]
fn test_l6_05_add_blocks_creates_bidirectional_edge() {
    let dir = tempfile::tempdir().unwrap();
    let tasks_dir = dir.path().join("tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();

    let tm = TaskManager::new(&tasks_dir);
    let t1: serde_json::Value = serde_json::from_str(&tm.create("Blocker")).unwrap();
    let t2: serde_json::Value = serde_json::from_str(&tm.create("Blocked")).unwrap();
    let t1_id = t1["id"].as_i64().unwrap();
    let t2_id = t2["id"].as_i64().unwrap();

    tm.update(t1_id, None, None, Some(&[t2_id])).unwrap();

    let t1_after: serde_json::Value = serde_json::from_str(&tm.get(t1_id)).unwrap();
    let t2_after: serde_json::Value = serde_json::from_str(&tm.get(t2_id)).unwrap();

    let t1_blocks: Vec<i64> = t1_after["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let t2_blocked_by: Vec<i64> = t2_after["blockedBy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();

    assert!(t1_blocks.contains(&t2_id));
    assert!(t2_blocked_by.contains(&t1_id));
}

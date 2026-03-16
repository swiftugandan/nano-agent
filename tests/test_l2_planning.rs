use nano_agent::planning::*;

// ---------------------------------------------------------------------------
// TestTodoManager
// ---------------------------------------------------------------------------

#[test]
fn test_l2_01_rejects_two_in_progress() {
    let mut tm = TodoManager::new();
    let result = tm.update(vec![
        TodoItem {
            id: "1".to_string(),
            text: "Task A".to_string(),
            status: "in_progress".to_string(),
        },
        TodoItem {
            id: "2".to_string(),
            text: "Task B".to_string(),
            status: "in_progress".to_string(),
        },
    ]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.to_lowercase().contains("in_progress"));
}

#[test]
fn test_l2_02_allows_exactly_one_in_progress() {
    let mut tm = TodoManager::new();
    let result = tm.update(vec![
        TodoItem {
            id: "1".to_string(),
            text: "Task A".to_string(),
            status: "in_progress".to_string(),
        },
        TodoItem {
            id: "2".to_string(),
            text: "Task B".to_string(),
            status: "pending".to_string(),
        },
    ]);
    assert!(result.is_ok());
    assert_eq!(tm.items.len(), 2);
}

#[test]
fn test_l2_03_rejects_empty_content() {
    let mut tm = TodoManager::new();
    let result = tm.update(vec![TodoItem {
        id: "1".to_string(),
        text: "".to_string(),
        status: "pending".to_string(),
    }]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("text"));
}

#[test]
fn test_l2_04_rejects_invalid_status() {
    let mut tm = TodoManager::new();
    let result = tm.update(vec![TodoItem {
        id: "1".to_string(),
        text: "Valid task".to_string(),
        status: "invalid_status".to_string(),
    }]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("status"));
}

// ---------------------------------------------------------------------------
// TestNagPolicy
// ---------------------------------------------------------------------------

#[test]
fn test_l2_05_nag_fires_after_threshold() {
    let mut nag = NagPolicy::new(3);
    for _ in 0..3 {
        nag.tick();
    }
    assert!(nag.should_inject());
}

#[test]
fn test_l2_06_nag_resets_when_todo_used() {
    let mut nag = NagPolicy::new(3);
    nag.tick();
    nag.tick();
    assert_eq!(nag.rounds_since_todo, 2);
    // Simulate todo tool usage -> reset
    nag.reset();
    assert_eq!(nag.rounds_since_todo, 0);
    assert!(!nag.should_inject());
}

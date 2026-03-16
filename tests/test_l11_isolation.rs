use nano_agent::isolation::*;
use nano_agent::tasks::TaskManager;
use std::process::Command;

fn setup_git_repo(path: &std::path::Path) {
    Command::new("git")
        .args(["init"])
        .current_dir(path)
        .output()
        .unwrap();
    Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(path)
        .output()
        .unwrap();
    Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(path)
        .output()
        .unwrap();
    std::fs::write(path.join("README.md"), "# Test repo\n").unwrap();
    Command::new("git")
        .args(["add", "."])
        .current_dir(path)
        .output()
        .unwrap();
    Command::new("git")
        .args(["commit", "-m", "init"])
        .current_dir(path)
        .output()
        .unwrap();
}

fn make_manager(
    repo_root: &std::path::Path,
) -> (
    WorktreeManager<'static>,
    &'static TaskManager,
    &'static EventBus,
) {
    let tasks_dir = repo_root.join(".tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();
    let tasks: &'static TaskManager = Box::leak(Box::new(TaskManager::new(&tasks_dir)));
    let events_path = repo_root.join(".worktrees").join("events.jsonl");
    let events: &'static EventBus = Box::leak(Box::new(EventBus::new(&events_path)));
    let wm = WorktreeManager::new(repo_root, tasks, events);
    (wm, tasks, events)
}

#[test]
fn test_l11_01_create_produces_isolated_directory_and_branch() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    setup_git_repo(&repo);

    let (wm, _tasks, _events) = make_manager(&repo);
    let result = wm.create("feature-x");
    assert!(result.is_ok());

    // Directory exists
    assert!(repo.join(".worktrees").join("feature-x").exists());

    // Branch exists
    let r = Command::new("git")
        .args(["branch", "--list", "wt/feature-x"])
        .current_dir(&repo)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&r.stdout);
    assert!(stdout.contains("wt/feature-x"));

    // Index entry
    let index_content =
        std::fs::read_to_string(repo.join(".worktrees").join("index.json")).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_content).unwrap();
    let names: Vec<&str> = index["worktrees"]
        .as_array()
        .unwrap()
        .iter()
        .map(|wt| wt["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"feature-x"));
}

#[test]
fn test_l11_02_task_binding_is_bidirectional() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    setup_git_repo(&repo);

    let (wm, tasks, _events) = make_manager(&repo);
    let t: serde_json::Value = serde_json::from_str(&tasks.create("Feature X")).unwrap();
    let task_id = t["id"].as_i64().unwrap();

    wm.create_with_task("feature-x", Some(task_id)).unwrap();

    // Check worktree index
    let index_content =
        std::fs::read_to_string(repo.join(".worktrees").join("index.json")).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_content).unwrap();
    let wt_entry = index["worktrees"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["name"].as_str() == Some("feature-x"))
        .unwrap();
    assert_eq!(wt_entry["task_id"].as_i64().unwrap(), task_id);

    // Check task
    let task: serde_json::Value = serde_json::from_str(&tasks.get(task_id)).unwrap();
    assert_eq!(task["worktree"].as_str().unwrap(), "feature-x");
}

#[test]
fn test_l11_03_binding_pending_task_advances_to_in_progress() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    setup_git_repo(&repo);

    let (wm, tasks, _events) = make_manager(&repo);
    let t: serde_json::Value = serde_json::from_str(&tasks.create("Pending task")).unwrap();
    assert_eq!(t["status"].as_str().unwrap(), "pending");

    wm.create_with_task("work-on-it", Some(t["id"].as_i64().unwrap()))
        .unwrap();

    let task: serde_json::Value =
        serde_json::from_str(&tasks.get(t["id"].as_i64().unwrap())).unwrap();
    assert_eq!(task["status"].as_str().unwrap(), "in_progress");
}

#[test]
fn test_l11_04_remove_with_complete_task() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    setup_git_repo(&repo);

    let (wm, tasks, _events) = make_manager(&repo);
    let t: serde_json::Value = serde_json::from_str(&tasks.create("Complete me")).unwrap();
    let task_id = t["id"].as_i64().unwrap();
    wm.create_with_task("finish-it", Some(task_id)).unwrap();

    wm.remove_with_options("finish-it", true).unwrap();

    // Task completed
    let task: serde_json::Value = serde_json::from_str(&tasks.get(task_id)).unwrap();
    assert_eq!(task["status"].as_str().unwrap(), "completed");
    assert_eq!(task["worktree"].as_str().unwrap(), "");

    // Worktree index status
    let index_content =
        std::fs::read_to_string(repo.join(".worktrees").join("index.json")).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_content).unwrap();
    let wt_entry = index["worktrees"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["name"].as_str() == Some("finish-it"))
        .unwrap();
    assert_eq!(wt_entry["status"].as_str().unwrap(), "removed");
}

#[test]
fn test_l11_06_name_validation_rejects_escape() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    setup_git_repo(&repo);

    let (wm, _tasks, _events) = make_manager(&repo);
    let result = wm.create("../escape");
    assert!(result.is_err());
}

#[test]
fn test_l11_05_events_emitted_for_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    setup_git_repo(&repo);

    let tasks_dir = repo.join(".tasks");
    std::fs::create_dir_all(&tasks_dir).unwrap();
    let tasks = Box::leak(Box::new(TaskManager::new(&tasks_dir)));
    let events_path = repo.join(".worktrees").join("events.jsonl");
    let events = Box::leak(Box::new(EventBus::new(&events_path)));
    let wm = WorktreeManager::new(&repo, tasks, events);

    wm.create("evented").unwrap();
    wm.remove("evented").unwrap();

    let event_log = std::fs::read_to_string(&events_path).unwrap();
    let event_types: Vec<String> = event_log
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .map(|evt| evt["event"].as_str().unwrap_or("").to_string())
        .collect();

    assert!(event_types.contains(&"worktree.create.before".to_string()));
    assert!(event_types.contains(&"worktree.create.after".to_string()));
    assert!(event_types.contains(&"worktree.remove.before".to_string()));
    assert!(event_types.contains(&"worktree.remove.after".to_string()));
}

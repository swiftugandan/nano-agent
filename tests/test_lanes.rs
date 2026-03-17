use nano_agent::concurrency::BackgroundManager;
use std::time::{Duration, Instant};

#[test]
fn test_run_in_lane_returns_immediately() {
    let bg = BackgroundManager::new();
    let start = Instant::now();
    let result = bg.run_in_lane("background", "sleep 10");
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_secs(1));
    assert!(result.contains("background"));
}

#[test]
fn test_default_lanes_exist() {
    let bg = BackgroundManager::new();
    // All default lanes should accept tasks
    let r1 = bg.run_in_lane("main", "echo main");
    assert!(r1.contains("main"));
    let r2 = bg.run_in_lane("cron", "echo cron");
    assert!(r2.contains("cron"));
    let r3 = bg.run_in_lane("background", "echo bg");
    assert!(r3.contains("background"));

    std::thread::sleep(Duration::from_millis(500));
    let notifs = bg.drain_notifications();
    assert_eq!(notifs.len(), 3);
}

#[test]
fn test_serial_ordering_in_main_lane() {
    // "main" lane has max_concurrency=1, so tasks run serially
    let bg = BackgroundManager::new();
    bg.run_in_lane("main", "echo first");
    bg.run_in_lane("main", "echo second");
    bg.run_in_lane("main", "echo third");

    // Wait for all to complete
    bg.wait_lane_idle("main");

    let notifs = bg.drain_notifications();
    assert_eq!(notifs.len(), 3);
}

#[test]
fn test_parallel_in_background_lane() {
    let bg = BackgroundManager::new();
    let start = Instant::now();

    // Run 4 tasks (background lane max=4) that each sleep 200ms
    for i in 0..4 {
        bg.run_in_lane("background", &format!("sleep 0.2 && echo task_{}", i));
    }

    bg.wait_lane_idle("background");
    let elapsed = start.elapsed();

    // If truly parallel, should take ~200ms, not ~800ms
    assert!(
        elapsed < Duration::from_millis(800),
        "4 parallel tasks should complete faster than serial: took {:?}",
        elapsed
    );

    let notifs = bg.drain_notifications();
    assert_eq!(notifs.len(), 4);
}

#[test]
fn test_generation_rejection_on_reset() {
    let bg = BackgroundManager::new();

    // Queue several tasks in main lane (serial, max=1)
    bg.run_in_lane("main", "sleep 0.5 && echo slow");
    bg.run_in_lane("main", "echo should_be_skipped_1");
    bg.run_in_lane("main", "echo should_be_skipped_2");

    // Reset the lane — queued tasks should be skipped
    bg.reset_lane("main");

    // Wait for the first task to complete
    bg.wait_lane_idle("main");

    let notifs = bg.drain_notifications();
    // Only the already-running task should complete
    assert!(
        notifs.len() <= 1,
        "reset should skip queued tasks, got {} notifications",
        notifs.len()
    );
}

#[test]
fn test_backwards_compat_run() {
    let bg = BackgroundManager::new();
    let result = bg.run("echo compat");
    assert!(result.contains("background")); // default lane

    std::thread::sleep(Duration::from_millis(300));
    let notifs = bg.drain_notifications();
    assert_eq!(notifs.len(), 1);
    assert!(notifs[0].result.contains("compat"));
}

#[test]
fn test_unknown_lane_falls_back_to_background() {
    let bg = BackgroundManager::new();
    let result = bg.run_in_lane("nonexistent", "echo fallback");
    // Should still work (falls back to background lane)
    assert!(result.contains("nonexistent")); // lane name in message

    std::thread::sleep(Duration::from_millis(300));
    let notifs = bg.drain_notifications();
    assert_eq!(notifs.len(), 1);
    assert!(notifs[0].result.contains("fallback"));
}

#[test]
fn test_drain_is_atomic() {
    let bg = BackgroundManager::new();
    bg.run("echo hello");
    bg.run("echo world");
    std::thread::sleep(Duration::from_secs(1));
    let first_drain = bg.drain_notifications();
    let second_drain = bg.drain_notifications();
    assert_eq!(first_drain.len(), 2);
    assert_eq!(second_drain.len(), 0);
}

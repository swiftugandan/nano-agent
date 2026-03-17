use nano_agent::heartbeat::{should_fire, CronScheduler, HeartbeatManager};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Cron parsing tests
// ---------------------------------------------------------------------------

#[test]
fn test_every_minute_matches() {
    let now = chrono::Local::now();
    // "* * * * *" should always match (if no recent fire)
    assert!(should_fire("* * * * *", now, None));
}

#[test]
fn test_specific_minute() {
    use chrono::TimeZone;
    let now = chrono::Local
        .with_ymd_and_hms(2026, 3, 16, 10, 30, 0)
        .unwrap();

    assert!(should_fire("30 * * * *", now, None));
    assert!(!should_fire("15 * * * *", now, None));
}

#[test]
fn test_every_n_minutes() {
    use chrono::TimeZone;
    let now_0 = chrono::Local
        .with_ymd_and_hms(2026, 3, 16, 10, 0, 0)
        .unwrap();
    let now_5 = chrono::Local
        .with_ymd_and_hms(2026, 3, 16, 10, 5, 0)
        .unwrap();
    let now_7 = chrono::Local
        .with_ymd_and_hms(2026, 3, 16, 10, 7, 0)
        .unwrap();

    assert!(should_fire("*/5 * * * *", now_0, None));
    assert!(should_fire("*/5 * * * *", now_5, None));
    assert!(!should_fire("*/5 * * * *", now_7, None));
}

#[test]
fn test_prevents_double_fire_within_minute() {
    let now = chrono::Local::now();
    let last = Some(now - chrono::Duration::seconds(30));
    assert!(!should_fire("* * * * *", now, last));
}

#[test]
fn test_fires_after_minute_gap() {
    let now = chrono::Local::now();
    let last = Some(now - chrono::Duration::seconds(90));
    assert!(should_fire("* * * * *", now, last));
}

#[test]
fn test_invalid_cron_does_not_fire() {
    let now = chrono::Local::now();
    assert!(!should_fire("invalid", now, None));
    assert!(!should_fire("1 2 3", now, None)); // too few fields
}

#[test]
fn test_specific_hour_and_minute() {
    use chrono::TimeZone;
    let now = chrono::Local
        .with_ymd_and_hms(2026, 3, 16, 14, 30, 0)
        .unwrap();

    assert!(should_fire("30 14 * * *", now, None));
    assert!(!should_fire("30 15 * * *", now, None));
    assert!(!should_fire("0 14 * * *", now, None));
}

// ---------------------------------------------------------------------------
// Scheduler tests
// ---------------------------------------------------------------------------

#[test]
fn test_add_and_list_entries() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().join("cron.json");

    let scheduler = CronScheduler::new(&config);
    scheduler.add("test-job", "*/5 * * * *", "Run tests");

    let list = scheduler.list();
    assert!(list.contains("test-job"));
    assert!(list.contains("*/5 * * * *"));
    assert!(list.contains("Run tests"));
}

#[test]
fn test_remove_entry() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().join("cron.json");

    let scheduler = CronScheduler::new(&config);
    scheduler.add("job1", "* * * * *", "prompt1");
    scheduler.add("job2", "*/5 * * * *", "prompt2");

    let result = scheduler.remove("job1");
    assert!(result.contains("removed"));

    let list = scheduler.list();
    assert!(!list.contains("job1"));
    assert!(list.contains("job2"));
}

#[test]
fn test_remove_nonexistent() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().join("cron.json");

    let scheduler = CronScheduler::new(&config);
    let result = scheduler.remove("ghost");
    assert!(result.contains("not found"));
}

#[test]
fn test_persistence() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().join("cron.json");

    // Add entries
    {
        let scheduler = CronScheduler::new(&config);
        scheduler.add("persisted", "0 * * * *", "hourly check");
    }

    // Reload from file
    {
        let scheduler = CronScheduler::new(&config);
        let list = scheduler.list();
        assert!(list.contains("persisted"));
        assert!(list.contains("hourly check"));
    }
}

#[test]
fn test_tick_fires_matching_entries() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().join("cron.json");

    let scheduler = CronScheduler::new(&config);
    // "* * * * *" should always match
    scheduler.add("always", "* * * * *", "always fires");

    let events = scheduler.tick();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "always");
    assert_eq!(events[0].prompt, "always fires");
}

#[test]
fn test_tick_prevents_double_fire() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().join("cron.json");

    let scheduler = CronScheduler::new(&config);
    scheduler.add("once", "* * * * *", "fire once");

    let events1 = scheduler.tick();
    assert_eq!(events1.len(), 1);

    // Second tick within same minute should not fire
    let events2 = scheduler.tick();
    assert_eq!(events2.len(), 0);
}

#[test]
fn test_replace_existing_entry() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().join("cron.json");

    let scheduler = CronScheduler::new(&config);
    scheduler.add("job", "* * * * *", "original");
    scheduler.add("job", "*/5 * * * *", "updated");

    let list = scheduler.list();
    assert!(list.contains("updated"));
    assert!(!list.contains("original"));
}

// ---------------------------------------------------------------------------
// HeartbeatManager tests
// ---------------------------------------------------------------------------

#[test]
fn test_drain_events() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().join("cron.json");
    let hb = HeartbeatManager::new(&config);

    // Add an always-firing cron and tick manually via scheduler
    hb.scheduler().add("drain-test", "* * * * *", "test prompt");
    let _events = hb.scheduler().tick();

    // The HeartbeatManager's pending list is separate from direct tick(),
    // but we can verify empty drain
    let drained = hb.drain_events();
    assert!(drained.is_empty()); // tick() was called on scheduler directly, not via the timer thread
}

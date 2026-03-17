use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

// ---------------------------------------------------------------------------
// Cron entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronEntry {
    pub name: String,
    pub cron: String,       // "*/5 * * * *"
    pub prompt: String,
    pub enabled: bool,
}

// ---------------------------------------------------------------------------
// Heartbeat event (injected into conversation)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HeartbeatEvent {
    pub name: String,
    pub prompt: String,
    pub fired_at: String,
}

// ---------------------------------------------------------------------------
// Minimal cron parser
// ---------------------------------------------------------------------------

/// Check if a cron field matches a given value.
/// Supports: "*", "*/N", and specific numbers.
fn field_matches(field: &str, value: u32) -> bool {
    if field == "*" {
        return true;
    }
    if let Some(step) = field.strip_prefix("*/") {
        if let Ok(n) = step.parse::<u32>() {
            return n > 0 && value % n == 0;
        }
    }
    if let Ok(specific) = field.parse::<u32>() {
        return value == specific;
    }
    false
}

/// Check if a cron expression should fire at the given time.
/// Format: "minute hour day-of-month month day-of-week"
pub fn should_fire(
    cron_expr: &str,
    now: chrono::DateTime<chrono::Local>,
    last: Option<chrono::DateTime<chrono::Local>>,
) -> bool {
    // Don't fire more than once per minute
    if let Some(last_fired) = last {
        if now.signed_duration_since(last_fired).num_seconds() < 60 {
            return false;
        }
    }

    let fields: Vec<&str> = cron_expr.split_whitespace().collect();
    if fields.len() != 5 {
        return false;
    }

    use chrono::Datelike;
    use chrono::Timelike;
    let minute = now.minute();
    let hour = now.hour();
    let day = now.day();
    let month = now.month();
    let weekday = now.weekday().num_days_from_sunday(); // 0=Sun

    field_matches(fields[0], minute)
        && field_matches(fields[1], hour)
        && field_matches(fields[2], day)
        && field_matches(fields[3], month)
        && field_matches(fields[4], weekday)
}

// ---------------------------------------------------------------------------
// CronScheduler
// ---------------------------------------------------------------------------

pub struct CronScheduler {
    entries: Arc<Mutex<Vec<CronEntry>>>,
    last_fired: Arc<Mutex<HashMap<String, chrono::DateTime<chrono::Local>>>>,
    config_path: PathBuf,
}

impl CronScheduler {
    pub fn new(config_path: &Path) -> Self {
        let entries = if config_path.exists() {
            let data = std::fs::read_to_string(config_path).unwrap_or_default();
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        Self {
            entries: Arc::new(Mutex::new(entries)),
            last_fired: Arc::new(Mutex::new(HashMap::new())),
            config_path: config_path.to_path_buf(),
        }
    }

    pub fn add(&self, name: &str, cron: &str, prompt: &str) -> String {
        let mut entries = self.entries.lock().unwrap();

        // Replace if name already exists
        entries.retain(|e| e.name != name);
        entries.push(CronEntry {
            name: name.to_string(),
            cron: cron.to_string(),
            prompt: prompt.to_string(),
            enabled: true,
        });

        self.persist_locked(&entries);
        format!("Cron '{}' added: {}", name, cron)
    }

    pub fn remove(&self, name: &str) -> String {
        let mut entries = self.entries.lock().unwrap();
        let before = entries.len();
        entries.retain(|e| e.name != name);
        if entries.len() == before {
            return format!("Error: cron '{}' not found", name);
        }
        self.persist_locked(&entries);
        format!("Cron '{}' removed", name)
    }

    pub fn list(&self) -> String {
        let entries = self.entries.lock().unwrap();
        if entries.is_empty() {
            return "No cron entries.".to_string();
        }
        entries
            .iter()
            .map(|e| {
                format!(
                    "  {} [{}] {}: {}",
                    if e.enabled { "+" } else { "-" },
                    e.cron,
                    e.name,
                    e.prompt
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Check all entries and return events that should fire now.
    pub fn tick(&self) -> Vec<HeartbeatEvent> {
        let entries = self.entries.lock().unwrap();
        if entries.is_empty() {
            return Vec::new();
        }
        let now = chrono::Local::now();
        let mut last_fired = self.last_fired.lock().unwrap();
        let mut events = Vec::new();

        for entry in entries.iter() {
            if !entry.enabled {
                continue;
            }
            let last = last_fired.get(&entry.name).copied();
            if should_fire(&entry.cron, now, last) {
                events.push(HeartbeatEvent {
                    name: entry.name.clone(),
                    prompt: entry.prompt.clone(),
                    fired_at: now.format("%Y-%m-%d %H:%M:%S").to_string(),
                });
                last_fired.insert(entry.name.clone(), now);
            }
        }

        events
    }

    fn persist_locked(&self, entries: &[CronEntry]) {
        if let Ok(json) = serde_json::to_string_pretty(entries) {
            std::fs::write(&self.config_path, json).ok();
        }
    }
}

// ---------------------------------------------------------------------------
// HeartbeatManager
// ---------------------------------------------------------------------------

pub struct HeartbeatManager {
    scheduler: Arc<CronScheduler>,
    pending: Arc<Mutex<Vec<HeartbeatEvent>>>,
}

impl HeartbeatManager {
    pub fn new(config_path: &Path) -> Self {
        Self {
            scheduler: Arc::new(CronScheduler::new(config_path)),
            pending: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Start the background timer thread that checks cron entries every 30s.
    pub fn start(&self) {
        let scheduler = Arc::clone(&self.scheduler);
        let pending = Arc::clone(&self.pending);

        thread::spawn(move || {
            loop {
                thread::sleep(std::time::Duration::from_secs(30));
                let events = scheduler.tick();
                if !events.is_empty() {
                    let mut p = pending.lock().unwrap();
                    p.extend(events);
                }
            }
        });
    }

    /// Drain all pending heartbeat events.
    pub fn drain_events(&self) -> Vec<HeartbeatEvent> {
        let mut p = self.pending.lock().unwrap();
        p.drain(..).collect()
    }

    /// Add a cron entry.
    pub fn add_cron(&self, name: &str, cron: &str, prompt: &str) -> String {
        self.scheduler.add(name, cron, prompt)
    }

    /// Remove a cron entry by name.
    pub fn remove_cron(&self, name: &str) -> String {
        self.scheduler.remove(name)
    }

    /// List all cron entries.
    pub fn list_crons(&self) -> String {
        self.scheduler.list()
    }

    /// Access the underlying scheduler (for testing).
    pub fn scheduler(&self) -> &CronScheduler {
        &self.scheduler
    }
}

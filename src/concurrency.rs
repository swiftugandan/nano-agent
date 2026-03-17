use std::collections::{HashMap, VecDeque};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LaneConfig {
    pub name: String,
    pub max_concurrency: usize,
}

#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub status: String,
    pub result: String,
    pub command: String,
    pub lane: String,
}

#[derive(Debug, Clone)]
pub struct Notification {
    pub task_id: String,
    pub status: String,
    pub command: String,
    pub result: String,
}

struct QueuedTask {
    task_id: String,
    generation: u64,
    command: String,
}

// Lane name constants
pub const LANE_MAIN: &str = "main";
pub const LANE_CRON: &str = "cron";
pub const LANE_BACKGROUND: &str = "background";

// ---------------------------------------------------------------------------
// Lane
// ---------------------------------------------------------------------------

struct Lane {
    config: LaneConfig,
    generation: AtomicU64,
    queue: Mutex<VecDeque<QueuedTask>>,
    active_count: AtomicU64,
    idle_pair: (Mutex<bool>, Condvar),
    // Shared references needed for pump()
    all_tasks: Arc<Mutex<HashMap<String, TaskInfo>>>,
    notification_queue: Arc<Mutex<Vec<Notification>>>,
}

impl Lane {
    fn new(
        config: LaneConfig,
        all_tasks: Arc<Mutex<HashMap<String, TaskInfo>>>,
        notification_queue: Arc<Mutex<Vec<Notification>>>,
    ) -> Self {
        Self {
            config,
            generation: AtomicU64::new(0),
            queue: Mutex::new(VecDeque::new()),
            active_count: AtomicU64::new(0),
            idle_pair: (Mutex::new(true), Condvar::new()),
            all_tasks,
            notification_queue,
        }
    }

    fn submit(self: &Arc<Self>, task_id: String, command: String) {
        let gen = self.generation.load(Ordering::Acquire);
        let mut q = self.queue.lock().unwrap();
        q.push_back(QueuedTask {
            task_id,
            generation: gen,
            command,
        });
        drop(q);
        self.pump();
    }

    fn pump(self: &Arc<Self>) {
        loop {
            let current_active = self.active_count.load(Ordering::Acquire) as usize;
            if current_active >= self.config.max_concurrency {
                break;
            }

            let task = {
                let mut q = self.queue.lock().unwrap();
                loop {
                    match q.pop_front() {
                        Some(t) => {
                            let current_gen = self.generation.load(Ordering::Acquire);
                            if t.generation < current_gen {
                                // Skip stale tasks
                                continue;
                            }
                            break Some(t);
                        }
                        None => break None,
                    }
                }
            };

            match task {
                Some(queued) => {
                    self.active_count.fetch_add(1, Ordering::Release);
                    // Mark not idle
                    {
                        let mut idle = self.idle_pair.0.lock().unwrap();
                        *idle = false;
                    }
                    let lane = Arc::clone(self);
                    let tid = queued.task_id.clone();
                    let cmd = queued.command.clone();

                    thread::spawn(move || {
                        let (status, output) = execute_command(&cmd);

                        // Update task entry
                        if let Ok(mut t) = lane.all_tasks.lock() {
                            if let Some(info) = t.get_mut(&tid) {
                                info.status = status.clone();
                                info.result = output.clone();
                            }
                        }

                        // Push notification and evict completed task atomically
                        {
                            lane.notification_queue.lock().unwrap().push(Notification {
                                task_id: tid.clone(),
                                status,
                                command: truncate_str(&cmd, 80).to_string(),
                                result: truncate_str(&output, 500).to_string(),
                            });
                            // Evict completed task from all_tasks to prevent unbounded growth
                            if let Ok(mut tasks) = lane.all_tasks.lock() {
                                tasks.remove(&tid);
                            }
                        }

                        // Decrement active, check idle, and pump — all under idle lock
                        // to prevent race between active_count check and queue check.
                        {
                            let mut idle = lane.idle_pair.0.lock().unwrap();
                            lane.active_count.fetch_sub(1, Ordering::Release);
                            let active = lane.active_count.load(Ordering::Acquire);
                            let queue_empty = lane.queue.lock().unwrap().is_empty();
                            if active == 0 && queue_empty {
                                *idle = true;
                                lane.idle_pair.1.notify_all();
                            }
                        }

                        lane.pump();
                    });
                }
                None => break,
            }
        }
    }

    fn reset(&self) {
        self.generation.fetch_add(1, Ordering::Release);
        self.queue.lock().unwrap().clear();
    }

    fn wait_idle(&self) {
        let (lock, cvar) = &self.idle_pair;
        let mut idle = lock.lock().unwrap();
        while !*idle {
            idle = cvar.wait(idle).unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// BackgroundManager
// ---------------------------------------------------------------------------

pub struct BackgroundManager {
    lanes: HashMap<String, Arc<Lane>>,
    all_tasks: Arc<Mutex<HashMap<String, TaskInfo>>>,
    notification_queue: Arc<Mutex<Vec<Notification>>>,
}

impl Default for BackgroundManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundManager {
    pub fn new() -> Self {
        let all_tasks = Arc::new(Mutex::new(HashMap::new()));
        let notification_queue = Arc::new(Mutex::new(Vec::new()));

        let default_lanes = vec![
            LaneConfig {
                name: LANE_MAIN.to_string(),
                max_concurrency: 1,
            },
            LaneConfig {
                name: LANE_CRON.to_string(),
                max_concurrency: 1,
            },
            LaneConfig {
                name: LANE_BACKGROUND.to_string(),
                max_concurrency: 4,
            },
        ];

        let mut lanes = HashMap::new();
        for config in default_lanes {
            let name = config.name.clone();
            lanes.insert(
                name,
                Arc::new(Lane::new(
                    config,
                    Arc::clone(&all_tasks),
                    Arc::clone(&notification_queue),
                )),
            );
        }

        Self {
            lanes,
            all_tasks,
            notification_queue,
        }
    }

    /// Run a command in a named lane. Returns a task ID immediately.
    pub fn run_in_lane(&self, lane: &str, command: &str) -> String {
        let task_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

        self.all_tasks.lock().unwrap().insert(
            task_id.clone(),
            TaskInfo {
                status: "running".to_string(),
                result: String::new(),
                command: command.to_string(),
                lane: lane.to_string(),
            },
        );

        if let Some(l) = self.lanes.get(lane) {
            l.submit(task_id.clone(), command.to_string());
        } else {
            // Fallback: use background lane
            if let Some(l) = self.lanes.get(LANE_BACKGROUND) {
                l.submit(task_id.clone(), command.to_string());
            }
        }

        format!(
            "Background task {} started in '{}': {}",
            task_id,
            lane,
            truncate_str(command, 80)
        )
    }

    /// Backwards-compatible: run in the "background" lane.
    pub fn run(&self, command: &str) -> String {
        self.run_in_lane(LANE_BACKGROUND, command)
    }

    /// Check status of one task.
    pub fn check(&self, task_id: &str) -> String {
        let tasks = self.all_tasks.lock().unwrap();
        match tasks.get(task_id) {
            Some(t) => format!(
                "[{}] {}\n{}",
                t.status,
                truncate_str(&t.command, 60),
                if t.result.is_empty() {
                    "(running)"
                } else {
                    &t.result
                }
            ),
            None => format!("Error: Unknown task {}", task_id),
        }
    }

    /// Return and clear all pending completion notifications.
    pub fn drain_notifications(&self) -> Vec<Notification> {
        let mut q = self.notification_queue.lock().unwrap();
        q.drain(..).collect()
    }

    /// Increment the lane generation, causing all queued tasks to be skipped.
    pub fn reset_lane(&self, lane: &str) {
        if let Some(l) = self.lanes.get(lane) {
            l.reset();
        }
    }

    /// Block until a lane has no active or queued tasks.
    pub fn wait_lane_idle(&self, lane: &str) {
        if let Some(l) = self.lanes.get(lane) {
            l.wait_idle();
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use crate::util::truncate_at_boundary as truncate_str;

fn execute_command(cmd: &str) -> (String, String) {
    match Command::new("sh").arg("-c").arg(cmd).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{}{}", stdout, stderr).trim().to_string();
            let result = if combined.is_empty() {
                "(no output)".to_string()
            } else {
                truncate_str(&combined, 50000).to_string()
            };
            ("completed".to_string(), result)
        }
        Err(e) => ("error".to_string(), format!("Error: {}", e)),
    }
}

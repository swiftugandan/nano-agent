use crate::channels::ChannelManager;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Delivery item
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeliveryItem {
    pub id: String,
    pub channel: String,
    pub peer_id: String,
    pub payload: String,
    pub attempts: u32,
    pub max_attempts: u32,
    pub next_attempt_epoch: u64,
}

// ---------------------------------------------------------------------------
// Write-ahead delivery queue
// ---------------------------------------------------------------------------

pub struct DeliveryQueue {
    queue_dir: PathBuf,
    dead_letter_dir: PathBuf,
}

impl DeliveryQueue {
    pub fn new(base_dir: &Path) -> Self {
        let queue_dir = base_dir.join("queue");
        let dead_letter_dir = base_dir.join("dead_letter");
        fs::create_dir_all(&queue_dir).ok();
        fs::create_dir_all(&dead_letter_dir).ok();
        Self {
            queue_dir,
            dead_letter_dir,
        }
    }

    /// Enqueue a delivery item with atomic file write (temp → fsync → rename).
    pub fn enqueue(&self, item: &DeliveryItem) -> Result<(), String> {
        let filename = format!("{}.json", item.id);
        let final_path = self.queue_dir.join(&filename);
        let temp_path = self.queue_dir.join(format!(".{}.tmp", item.id));

        let data = serde_json::to_string_pretty(item).map_err(|e| e.to_string())?;

        // Atomic write: temp → fsync → rename
        let mut file = fs::File::create(&temp_path).map_err(|e| e.to_string())?;
        file.write_all(data.as_bytes()).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);

        fs::rename(&temp_path, &final_path).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load all pending delivery items from the queue directory.
    pub fn load_pending(&self) -> Vec<DeliveryItem> {
        let mut items = Vec::new();
        let dir = match fs::read_dir(&self.queue_dir) {
            Ok(d) => d,
            Err(_) => return items,
        };
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(item) = serde_json::from_str::<DeliveryItem>(&content) {
                    items.push(item);
                }
            }
        }
        items
    }

    /// Remove a successfully delivered item.
    pub fn remove(&self, id: &str) {
        let path = self.queue_dir.join(format!("{}.json", id));
        fs::remove_file(&path).ok();
    }

    /// Move a failed item to the dead letter directory.
    pub fn dead_letter(&self, item: &DeliveryItem) {
        let src = self.queue_dir.join(format!("{}.json", item.id));
        let dst = self.dead_letter_dir.join(format!("{}.json", item.id));
        // Write updated item to dead letter
        if let Ok(data) = serde_json::to_string_pretty(item) {
            fs::write(&dst, data).ok();
        }
        fs::remove_file(&src).ok();
    }

    /// Update an item in the queue (e.g., increment attempts).
    pub fn update(&self, item: &DeliveryItem) -> Result<(), String> {
        self.enqueue(item) // overwrite via atomic write
    }
}

// ---------------------------------------------------------------------------
// Delivery runner: background thread polling the queue
// ---------------------------------------------------------------------------

const DEFAULT_POLL_INTERVAL_MS: u64 = 2000;
const DEFAULT_BASE_DELAY_MS: u64 = 1000;
const DEFAULT_MAX_DELAY_MS: u64 = 60_000;

pub struct DeliveryRunner {
    queue: Arc<DeliveryQueue>,
    channel_manager: Arc<Mutex<ChannelManager>>,
    poll_interval_ms: u64,
    base_delay_ms: u64,
    max_delay_ms: u64,
}

impl DeliveryRunner {
    pub fn new(queue: Arc<DeliveryQueue>, channel_manager: Arc<Mutex<ChannelManager>>) -> Self {
        Self {
            queue,
            channel_manager,
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            base_delay_ms: DEFAULT_BASE_DELAY_MS,
            max_delay_ms: DEFAULT_MAX_DELAY_MS,
        }
    }

    /// Start the delivery runner in a background thread.
    pub fn start(self) {
        std::thread::spawn(move || loop {
            self.process_pending();
            std::thread::sleep(std::time::Duration::from_millis(self.poll_interval_ms));
        });
    }

    fn process_pending(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let items = self.queue.load_pending();
        for mut item in items {
            // Not yet time to retry?
            if item.next_attempt_epoch > now {
                continue;
            }

            // Attempt delivery
            let result = self.channel_manager.lock().unwrap().send(
                &item.channel,
                &item.peer_id,
                &item.payload,
            );

            match result {
                Ok(()) => {
                    self.queue.remove(&item.id);
                    eprintln!(
                        "[delivery] Delivered {} to {}:{}",
                        item.id, item.channel, item.peer_id
                    );
                }
                Err(e) => {
                    item.attempts += 1;
                    eprintln!(
                        "[delivery] Failed {} (attempt {}/{}): {}",
                        item.id, item.attempts, item.max_attempts, e
                    );

                    if item.attempts >= item.max_attempts {
                        eprintln!("[delivery] Dead-lettered {}", item.id);
                        self.queue.dead_letter(&item);
                    } else {
                        // Exponential backoff
                        let delay = self.compute_backoff(item.attempts);
                        item.next_attempt_epoch = now + (delay.as_secs());
                        self.queue.update(&item).ok();
                    }
                }
            }
        }
    }

    fn compute_backoff(&self, attempts: u32) -> std::time::Duration {
        let delay_ms = self.base_delay_ms * 2u64.pow(attempts.saturating_sub(1));
        let capped = delay_ms.min(self.max_delay_ms);
        std::time::Duration::from_millis(capped)
    }
}

// ---------------------------------------------------------------------------
// Convenience: create and enqueue a delivery
// ---------------------------------------------------------------------------

pub fn enqueue_delivery(
    queue: &DeliveryQueue,
    channel: &str,
    peer_id: &str,
    payload: &str,
    max_attempts: u32,
) -> Result<String, String> {
    let id = uuid::Uuid::new_v4().to_string();
    let item = DeliveryItem {
        id: id.clone(),
        channel: channel.to_string(),
        peer_id: peer_id.to_string(),
        payload: payload.to_string(),
        attempts: 0,
        max_attempts,
        next_attempt_epoch: 0,
    };
    queue.enqueue(&item)?;
    Ok(id)
}

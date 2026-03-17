use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// TF-IDF Memory Store
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub text: String,
    pub tags: Vec<String>,
    pub timestamp: String,
    pub score: f64,
}

pub struct MemoryStore {
    static_path: PathBuf,
    dynamic_dir: PathBuf,
}

fn stop_words() -> &'static HashSet<&'static str> {
    static INSTANCE: OnceLock<HashSet<&str>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        [
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "is", "it", "was", "are", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "shall", "can", "this", "that", "these", "those", "i", "you", "he", "she", "we",
            "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
            "not", "no", "so", "if", "as",
        ]
        .into_iter()
        .collect()
    })
}

impl MemoryStore {
    pub fn new(static_path: &Path, dynamic_dir: &Path) -> Self {
        fs::create_dir_all(dynamic_dir).ok();
        Self {
            static_path: static_path.to_path_buf(),
            dynamic_dir: dynamic_dir.to_path_buf(),
        }
    }

    /// Save a memory entry to today's JSONL file.
    pub fn save_memory(&self, text: &str, tags: &[String]) -> String {
        let now = chrono::Local::now();
        let date_str = now.format("%Y-%m-%d").to_string();
        let ts = now.format("%Y-%m-%dT%H:%M:%S").to_string();
        let path = self.dynamic_dir.join(format!("{}.jsonl", date_str));

        let entry = serde_json::json!({
            "text": text,
            "tags": tags,
            "timestamp": ts,
        });

        let mut line = serde_json::to_string(&entry).unwrap_or_default();
        line.push('\n');

        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| {
                use std::io::Write;
                f.write_all(line.as_bytes())
            })
            .ok();

        format!(
            "Memory saved: {} ({} tags)",
            &text[..text.len().min(80)],
            tags.len()
        )
    }

    /// Load all memory entries from dynamic JSONL files.
    fn load_all_entries(&self) -> Vec<MemoryEntry> {
        let mut entries = Vec::new();
        let dir = match fs::read_dir(&self.dynamic_dir) {
            Ok(d) => d,
            Err(_) => return entries,
        };
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if let Ok(content) = fs::read_to_string(&path) {
                for line in content.lines() {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                        entries.push(MemoryEntry {
                            text: val["text"].as_str().unwrap_or("").to_string(),
                            tags: val["tags"]
                                .as_array()
                                .map(|a| {
                                    a.iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect()
                                })
                                .unwrap_or_default(),
                            timestamp: val["timestamp"].as_str().unwrap_or("").to_string(),
                            score: 0.0,
                        });
                    }
                }
            }
        }
        entries
    }

    /// Recall top-k memories relevant to a query using TF-IDF scoring.
    pub fn recall(&self, query: &str, top_k: usize) -> Vec<MemoryEntry> {
        let mut entries = self.load_all_entries();
        if entries.is_empty() {
            return Vec::new();
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Build corpus: each entry's text + tags concatenated
        let docs: Vec<String> = entries
            .iter()
            .map(|e| {
                let mut s = e.text.clone();
                for tag in &e.tags {
                    s.push(' ');
                    s.push_str(tag);
                }
                s
            })
            .collect();

        let doc_tokens: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
        let n = doc_tokens.len() as f64;

        // Document frequency for each term
        let mut df: HashMap<&str, usize> = HashMap::new();
        for tokens in &doc_tokens {
            let unique: HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
            for term in unique {
                *df.entry(term).or_insert(0) += 1;
            }
        }

        // Score each document against the query
        let query_tf = term_frequency(&query_tokens);

        for (i, tokens) in doc_tokens.iter().enumerate() {
            let doc_tf = term_frequency(tokens);
            let mut score = 0.0;

            for (&term, &qtf) in &query_tf {
                let dtf = doc_tf.get(term).copied().unwrap_or(0.0);
                let idf = (n / (1.0 + *df.get(term).unwrap_or(&0) as f64)).ln();
                score += qtf * dtf * idf;
            }

            entries[i].score = score;
        }

        // Sort by score descending
        entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(top_k);

        // Filter out zero-score entries
        entries.retain(|e| e.score > 0.0);
        entries
    }

    /// Remove daily JSONL files older than `max_age_days`.
    pub fn evict_old(&self, max_age_days: u64) -> usize {
        let cutoff = chrono::Local::now() - chrono::Duration::days(max_age_days as i64);
        let cutoff_str = cutoff.format("%Y-%m-%d").to_string();
        let mut removed = 0;

        // Filenames are dates: "2025-01-15.jsonl"
        for path in crate::util::list_files_matching(&self.dynamic_dir, "", ".jsonl") {
            let name = match path.file_stem().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if name < cutoff_str.as_str() && fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        removed
    }

    /// Load the static MEMORY.md content.
    pub fn load_static(&self) -> String {
        fs::read_to_string(&self.static_path).unwrap_or_default()
    }
}

/// Tokenize: lowercase, split on non-alphanumeric, filter stop words and short tokens.
fn tokenize(text: &str) -> Vec<String> {
    let stops = stop_words();
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1 && !stops.contains(w))
        .map(String::from)
        .collect()
}

/// Compute normalized term frequency for a token list.
/// Returns borrowed keys to avoid cloning token strings.
fn term_frequency<'a>(tokens: &'a [String]) -> HashMap<&'a str, f64> {
    let mut counts: HashMap<&'a str, f64> = HashMap::new();
    for token in tokens {
        *counts.entry(token.as_str()).or_insert(0.0) += 1.0;
    }
    let max = counts.values().cloned().fold(0.0_f64, f64::max);
    if max > 0.0 {
        for val in counts.values_mut() {
            *val /= max;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello world, this is a test!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // stop words filtered
        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_term_frequency() {
        let tokens = vec!["hello".into(), "world".into(), "hello".into()];
        let tf = term_frequency(&tokens);
        assert_eq!(tf["hello"], 1.0); // max
        assert_eq!(tf["world"], 0.5); // half of max
    }

    #[test]
    fn test_save_and_recall() {
        let dir = tempfile::tempdir().unwrap();
        let static_path = dir.path().join("MEMORY.md");
        let dynamic_dir = dir.path().join("memories");
        let store = MemoryStore::new(&static_path, &dynamic_dir);

        store.save_memory(
            "The API endpoint uses JWT authentication",
            &["auth".into(), "api".into()],
        );
        store.save_memory("Database migrations run on startup", &["db".into()]);
        store.save_memory("Logging uses structured JSON format", &["logging".into()]);

        let results = store.recall("JWT authentication", 2);
        assert!(!results.is_empty());
        assert!(results[0].text.contains("JWT"));
    }

    #[test]
    fn test_evict_old() {
        let dir = tempfile::tempdir().unwrap();
        let static_path = dir.path().join("MEMORY.md");
        let dynamic_dir = dir.path().join("memories");
        let store = MemoryStore::new(&static_path, &dynamic_dir);

        // Create a "today" file (should survive)
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        fs::write(dynamic_dir.join(format!("{}.jsonl", today)), "{}").unwrap();

        // Create an "old" file (should be evicted)
        fs::write(dynamic_dir.join("2020-01-01.jsonl"), "{}").unwrap();

        // Create a non-JSONL file (should survive)
        fs::write(dynamic_dir.join("notes.txt"), "keep").unwrap();

        let removed = store.evict_old(90);
        assert_eq!(removed, 1);

        // Today's file still exists
        assert!(dynamic_dir.join(format!("{}.jsonl", today)).exists());
        // Old file was removed
        assert!(!dynamic_dir.join("2020-01-01.jsonl").exists());
        // Non-JSONL file untouched
        assert!(dynamic_dir.join("notes.txt").exists());
    }
}

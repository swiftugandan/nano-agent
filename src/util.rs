use std::env;
use std::path::{Path, PathBuf};

/// Require an environment variable, printing a friendly error and exiting if missing.
pub fn require_env(var_name: &str) -> String {
    env::var(var_name).unwrap_or_else(|_| {
        eprintln!("Error: {} environment variable is not set.", var_name);
        eprintln!("Set it with: export {}=your_key_here", var_name);
        std::process::exit(1);
    })
}

/// List files in a directory matching a name prefix and suffix, sorted.
pub fn list_files_matching(dir: &Path, prefix: &str, suffix: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(suffix))
            {
                paths.push(path);
            }
        }
    }
    paths.sort();
    paths
}

/// Truncate a string at a UTF-8 char boundary, avoiding panics on multi-byte sequences.
pub fn truncate_at_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

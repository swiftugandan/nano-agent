use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// MessageBus: JSONL inbox per teammate
// ---------------------------------------------------------------------------

pub struct MessageBus {
    pub dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboxMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    #[serde(rename = "from")]
    pub from: String,
    pub content: String,
    #[serde(default)]
    pub timestamp: f64,
    // Extra fields for protocol messages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approve: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
}

impl MessageBus {
    pub fn new(inbox_dir: &Path) -> Self {
        std::fs::create_dir_all(inbox_dir).ok();
        Self {
            dir: inbox_dir.to_path_buf(),
        }
    }

    pub fn send(&self, sender: &str, to: &str, content: &str) -> String {
        self.send_typed(sender, to, content, "message", None)
    }

    pub fn send_typed(
        &self,
        sender: &str,
        to: &str,
        content: &str,
        msg_type: &str,
        extra: Option<serde_json::Value>,
    ) -> String {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut msg = serde_json::json!({
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": timestamp,
        });

        if let Some(extra_val) = extra {
            if let Some(obj) = extra_val.as_object() {
                for (k, v) in obj {
                    msg[k] = v.clone();
                }
            }
        }

        let inbox_path = self.dir.join(format!("{}.jsonl", to));
        let line = serde_json::to_string(&msg).unwrap() + "\n";

        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&inbox_path)
            .unwrap();
        file.write_all(line.as_bytes()).unwrap();

        format!("Sent {} to {}", msg_type, to)
    }

    pub fn read_inbox(&self, name: &str) -> Vec<serde_json::Value> {
        let inbox_path = self.dir.join(format!("{}.jsonl", name));
        if !inbox_path.exists() {
            return Vec::new();
        }
        let content = std::fs::read_to_string(&inbox_path).unwrap_or_default();
        let messages: Vec<serde_json::Value> = content
            .lines()
            .filter(|l| !l.is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        // Drain: clear the file
        std::fs::write(&inbox_path, "").ok();
        messages
    }

    pub fn broadcast(&self, sender: &str, content: &str, teammates: &[String]) -> String {
        let mut count = 0;
        for name in teammates {
            if name != sender {
                self.send_typed(sender, name, content, "broadcast", None);
                count += 1;
            }
        }
        format!("Broadcast to {} teammates", count)
    }
}

// ---------------------------------------------------------------------------
// TeammateManager (simplified for contract tests)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamMember {
    pub name: String,
    pub role: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamConfig {
    pub team_name: String,
    pub members: Vec<TeamMember>,
}

pub struct TeammateManager {
    pub dir: PathBuf,
    pub config: TeamConfig,
}

impl TeammateManager {
    pub fn new(team_dir: &Path) -> Self {
        std::fs::create_dir_all(team_dir).ok();
        let config_path = team_dir.join("config.json");
        let config = if config_path.exists() {
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap_or_default())
                .unwrap_or(TeamConfig {
                    team_name: "default".to_string(),
                    members: Vec::new(),
                })
        } else {
            TeamConfig {
                team_name: "default".to_string(),
                members: Vec::new(),
            }
        };
        Self {
            dir: team_dir.to_path_buf(),
            config,
        }
    }

    pub fn find_member(&self, name: &str) -> Option<&TeamMember> {
        self.config.members.iter().find(|m| m.name == name)
    }

    pub fn find_member_mut(&mut self, name: &str) -> Option<&mut TeamMember> {
        self.config.members.iter_mut().find(|m| m.name == name)
    }

    pub fn spawn(&mut self, name: &str, role: &str, _prompt: &str) -> String {
        if let Some(member) = self.find_member_mut(name) {
            if member.status != "idle" && member.status != "shutdown" {
                return format!("Error: '{}' is currently {}", name, member.status);
            }
            member.status = "working".to_string();
            member.role = role.to_string();
        } else {
            self.config.members.push(TeamMember {
                name: name.to_string(),
                role: role.to_string(),
                status: "working".to_string(),
            });
        }
        self.save_config();
        format!("Spawned '{}' (role: {})", name, role)
    }

    pub fn list_all(&self) -> String {
        if self.config.members.is_empty() {
            return "No teammates.".to_string();
        }
        let mut lines = vec![format!("Team: {}", self.config.team_name)];
        for m in &self.config.members {
            lines.push(format!("  {} ({}): {}", m.name, m.role, m.status));
        }
        lines.join("\n")
    }

    pub fn member_names(&self) -> Vec<String> {
        self.config.members.iter().map(|m| m.name.clone()).collect()
    }

    fn save_config(&self) {
        let config_path = self.dir.join("config.json");
        let content = serde_json::to_string_pretty(&self.config).unwrap();
        std::fs::write(&config_path, content).ok();
    }
}

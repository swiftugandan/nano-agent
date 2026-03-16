use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// SkillLoader
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    pub tags: String,
}

#[derive(Debug, Clone)]
pub struct SkillEntry {
    pub meta: SkillMeta,
    pub body: String,
    pub path: String,
}

pub struct SkillLoader {
    pub skills: HashMap<String, SkillEntry>,
}

impl SkillLoader {
    pub fn new(skills_dir: &Path) -> Self {
        let mut loader = Self {
            skills: HashMap::new(),
        };
        loader.load_all(skills_dir);
        loader
    }

    fn load_all(&mut self, skills_dir: &Path) {
        if !skills_dir.exists() {
            return;
        }
        // Walk subdirectories looking for SKILL.md
        if let Ok(entries) = std::fs::read_dir(skills_dir) {
            let mut dirs: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            dirs.sort_by_key(|e| e.file_name());
            for entry in dirs {
                let skill_file = entry.path().join("SKILL.md");
                if skill_file.exists() {
                    if let Ok(text) = std::fs::read_to_string(&skill_file) {
                        let (meta, body) = Self::parse_frontmatter(&text);
                        let name = if meta.contains_key("name") {
                            meta["name"].clone()
                        } else {
                            entry.file_name().to_string_lossy().to_string()
                        };
                        let description = meta.get("description").cloned().unwrap_or_default();
                        let tags = meta.get("tags").cloned().unwrap_or_default();

                        self.skills.insert(
                            name.clone(),
                            SkillEntry {
                                meta: SkillMeta {
                                    name,
                                    description,
                                    tags,
                                },
                                body,
                                path: skill_file.to_string_lossy().to_string(),
                            },
                        );
                    }
                }
            }
        }
    }

    fn parse_frontmatter(text: &str) -> (HashMap<String, String>, String) {
        // Match --- delimited frontmatter
        if !text.starts_with("---\n") {
            return (HashMap::new(), text.to_string());
        }
        if let Some(end) = text[4..].find("\n---\n") {
            let frontmatter = &text[4..4 + end];
            let body = &text[4 + end + 5..]; // skip past \n---\n
            let mut meta = HashMap::new();
            for line in frontmatter.lines() {
                if let Some(colon_pos) = line.find(':') {
                    let key = line[..colon_pos].trim().to_string();
                    let val = line[colon_pos + 1..].trim().to_string();
                    meta.insert(key, val);
                }
            }
            (meta, body.trim().to_string())
        } else {
            (HashMap::new(), text.to_string())
        }
    }

    /// Layer 1: short descriptions for the system prompt.
    pub fn get_descriptions(&self) -> String {
        if self.skills.is_empty() {
            return "(no skills available)".to_string();
        }
        let mut lines: Vec<String> = Vec::new();
        let mut names: Vec<&String> = self.skills.keys().collect();
        names.sort();
        for name in names {
            let skill = &self.skills[name];
            let desc = if skill.meta.description.is_empty() {
                "No description"
            } else {
                &skill.meta.description
            };
            let mut line = format!("  - {}: {}", name, desc);
            if !skill.meta.tags.is_empty() {
                line.push_str(&format!(" [{}]", skill.meta.tags));
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    /// Layer 2: full skill body returned in tool_result.
    pub fn get_content(&self, name: &str) -> String {
        match self.skills.get(name) {
            Some(skill) => format!("<skill name=\"{}\">\n{}\n</skill>", name, skill.body),
            None => {
                let available: Vec<&String> = self.skills.keys().collect();
                format!(
                    "Error: Unknown skill '{}'. Available: {}",
                    name,
                    available
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
    }
}

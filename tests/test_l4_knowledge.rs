use nano_agent::knowledge::*;

fn create_skill(skills_dir: &std::path::Path, name: &str, description: &str, body: &str, tags: &str) {
    let skill_dir = skills_dir.join(name);
    std::fs::create_dir_all(&skill_dir).unwrap();
    let mut content = format!("---\nname: {}\ndescription: {}\n", name, description);
    if !tags.is_empty() {
        content.push_str(&format!("tags: {}\n", tags));
    }
    content.push_str(&format!("---\n{}", body));
    std::fs::write(skill_dir.join("SKILL.md"), content).unwrap();
}

#[test]
fn test_l4_01_descriptions_returns_compact_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let skills_dir = dir.path().join("skills");
    std::fs::create_dir_all(&skills_dir).unwrap();
    create_skill(
        &skills_dir,
        "test-skill",
        "A test skill",
        "This is a very long body content that should NOT appear in descriptions.",
        "",
    );
    let loader = SkillLoader::new(&skills_dir);
    let desc = loader.get_descriptions();
    assert!(desc.contains("test-skill"));
    assert!(desc.contains("A test skill"));
    assert!(!desc.contains("very long body content"));
}

#[test]
fn test_l4_02_load_returns_full_body_in_skill_tags() {
    let dir = tempfile::tempdir().unwrap();
    let skills_dir = dir.path().join("skills");
    std::fs::create_dir_all(&skills_dir).unwrap();
    create_skill(
        &skills_dir,
        "test-skill",
        "A test skill",
        "Full body content here.",
        "",
    );
    let loader = SkillLoader::new(&skills_dir);
    let content = loader.get_content("test-skill");
    assert!(content.contains("Full body content here."));
    assert!(content.contains("<skill"));
    assert!(content.contains("test-skill"));
}

#[test]
fn test_l4_03_unknown_skill_returns_error_with_available_names() {
    let dir = tempfile::tempdir().unwrap();
    let skills_dir = dir.path().join("skills");
    std::fs::create_dir_all(&skills_dir).unwrap();
    create_skill(&skills_dir, "real-skill", "A real skill", "Body.", "");
    let loader = SkillLoader::new(&skills_dir);
    let result = loader.get_content("nonexistent");
    assert!(result.contains("Error"));
    assert!(result.contains("real-skill"));
}

#[test]
fn test_l4_04_frontmatter_parsed_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let skills_dir = dir.path().join("skills");
    std::fs::create_dir_all(&skills_dir).unwrap();
    create_skill(
        &skills_dir,
        "parsed-skill",
        "Description here",
        "The body.",
        "tag1,tag2",
    );
    let loader = SkillLoader::new(&skills_dir);
    let skill = loader.skills.get("parsed-skill").unwrap();
    assert_eq!(skill.meta.name, "parsed-skill");
    assert_eq!(skill.meta.description, "Description here");
}

#[test]
fn test_l4_05_system_prompt_uses_descriptions_not_bodies() {
    let dir = tempfile::tempdir().unwrap();
    let skills_dir = dir.path().join("skills");
    std::fs::create_dir_all(&skills_dir).unwrap();
    for i in 0..3 {
        let body = "A ".repeat(500);
        create_skill(
            &skills_dir,
            &format!("skill-{}", i),
            &format!("Short {}", i),
            &body,
            "",
        );
    }
    let loader = SkillLoader::new(&skills_dir);
    let desc_tokens = loader.get_descriptions().len() / 4;
    let body_tokens: usize = loader.skills.values().map(|s| s.body.len() / 4).sum();
    assert!(desc_tokens < body_tokens);
}

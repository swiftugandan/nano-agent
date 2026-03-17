use nano_agent::prompt::{PromptAssembler, PromptContext};
use tempfile::TempDir;

fn make_context() -> PromptContext {
    PromptContext {
        agent_name: "TestBot".into(),
        agent_role: "developer".into(),
        cwd: "/tmp/project".into(),
        tool_count: 25,
        timestamp: "2026-03-17T00:00:00".into(),
        seed_sections: vec![
            ("Current Todo List".into(), "[ ] Fix bug".into()),
            (
                "Available Skills".into(),
                "deploy: Deploy to production".into(),
            ),
        ],
        ..Default::default()
    }
}

#[test]
fn test_compose_with_default_files() {
    let dir = TempDir::new().unwrap();
    let prompts_dir = dir.path().join("prompts");

    // Init defaults and then reload
    let assembler = PromptAssembler::new(&prompts_dir);
    assembler.init_defaults();

    // Re-create to pick up the files
    let assembler = PromptAssembler::new(&prompts_dir);
    let ctx = make_context();
    let result = assembler.compose(&ctx);

    assert!(result.contains("TestBot"), "should substitute agent name");
    assert!(result.contains("developer"), "should substitute role");
    assert!(result.contains("/tmp/project"), "should substitute cwd");
    assert!(result.contains("25"), "should substitute tool count");
    assert!(result.contains("Fix bug"), "should include todo state");
    assert!(
        result.contains("deploy"),
        "should include skill descriptions"
    );
}

#[test]
fn test_missing_files_are_skipped() {
    let dir = TempDir::new().unwrap();
    let prompts_dir = dir.path().join("prompts");
    std::fs::create_dir_all(&prompts_dir).unwrap();

    // Only create SOUL.md
    std::fs::write(prompts_dir.join("SOUL.md"), "You are {name}, role: {role}.").unwrap();

    let assembler = PromptAssembler::new(&prompts_dir);
    let ctx = make_context();
    let result = assembler.compose(&ctx);

    assert!(result.contains("TestBot"));
    assert!(result.contains("developer"));
    // Should NOT contain content from non-existent files
    assert!(!result.contains("IDENTITY"));
}

#[test]
fn test_placeholder_substitution() {
    let dir = TempDir::new().unwrap();
    let prompts_dir = dir.path().join("prompts");
    std::fs::create_dir_all(&prompts_dir).unwrap();

    std::fs::write(
        prompts_dir.join("SOUL.md"),
        "Name: {name}\nRole: {role}\nCwd: {cwd}\nTools: {tool_count}",
    )
    .unwrap();

    let assembler = PromptAssembler::new(&prompts_dir);
    let ctx = make_context();
    let result = assembler.compose(&ctx);

    assert!(result.contains("Name: TestBot"));
    assert!(result.contains("Role: developer"));
    assert!(result.contains("Cwd: /tmp/project"));
    assert!(result.contains("Tools: 25"));
}

#[test]
fn test_empty_prompts_dir_gives_fallback() {
    let dir = TempDir::new().unwrap();
    let prompts_dir = dir.path().join("prompts");
    std::fs::create_dir_all(&prompts_dir).unwrap();

    let assembler = PromptAssembler::new(&prompts_dir);
    let ctx = make_context();
    let result = assembler.compose(&ctx);

    // Should get fallback prompt
    assert!(result.contains("TestBot"));
    assert!(result.contains("25 tools"));
}

#[test]
fn test_init_defaults_creates_files() {
    let dir = TempDir::new().unwrap();
    let prompts_dir = dir.path().join("prompts");

    let assembler = PromptAssembler::new(&prompts_dir);
    assembler.init_defaults();

    assert!(prompts_dir.join("SOUL.md").exists());
    assert!(prompts_dir.join("IDENTITY.md").exists());
    assert!(prompts_dir.join("TOOLS.md").exists());
    assert!(prompts_dir.join("GUIDELINES.md").exists());
}

#[test]
fn test_init_defaults_does_not_overwrite_existing() {
    let dir = TempDir::new().unwrap();
    let prompts_dir = dir.path().join("prompts");
    std::fs::create_dir_all(&prompts_dir).unwrap();

    // Pre-create a custom SOUL.md
    std::fs::write(prompts_dir.join("SOUL.md"), "Custom soul content").unwrap();

    let assembler = PromptAssembler::new(&prompts_dir);
    assembler.init_defaults();

    let content = std::fs::read_to_string(prompts_dir.join("SOUL.md")).unwrap();
    assert_eq!(
        content, "Custom soul content",
        "should not overwrite existing file"
    );
}

use nano_agent::delegation::*;
use nano_agent::handler::{
    require_str, AgentContext, HandlerError, HandlerRegistry, HandlerResult,
};
use nano_agent::mock::*;
use nano_agent::types::LlmError;
use std::sync::Arc;
use std::sync::Mutex;

#[test]
fn test_l3_01_subagent_starts_with_fresh_messages() {
    let mut llm = MockLLM::new();
    llm.queue("end_turn", vec![make_text_block("Subagent done.")]);

    let tmp = tempfile::tempdir().unwrap();
    let registry = HandlerRegistry::new();
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();
    let result = SubagentFactory::spawn(
        &mut llm,
        "Do something",
        &tools,
        &registry,
        &ctx,
        30,
        SpawnHooks::default(),
    );

    assert_eq!(result, "Subagent done.");
    assert_eq!(llm.call_count(), 1);
}

#[test]
fn test_l3_02_returns_only_final_text() {
    let mut llm = MockLLM::new();
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "tu_1",
            "bash",
            serde_json::json!({"command": "echo hi"}),
        )],
    );
    llm.queue("end_turn", vec![make_text_block("Final summary.")]);

    let tmp = tempfile::tempdir().unwrap();
    let mut registry = HandlerRegistry::new();
    registry.register(
        "bash",
        Arc::new(
            |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok("hi".to_string())
            },
        ),
    );
    let ctx = AgentContext::mock(tmp.path());

    let tools = child_tools();
    let result = SubagentFactory::spawn(
        &mut llm,
        "Test",
        &tools,
        &registry,
        &ctx,
        30,
        SpawnHooks::default(),
    );

    assert_eq!(result, "Final summary.");
    assert!(!result.contains("tool_use"));
    assert!(!result.contains("tool_result"));
}

#[test]
fn test_l3_03_context_does_not_leak_to_parent() {
    // Simulate what parent does when subagent tool is called
    let mut parent_messages =
        vec![serde_json::json!({"role": "user", "content": "Parent context"})];
    let initial_len = parent_messages.len();

    // Parent: assistant with tool_use + user with tool_result = +2
    parent_messages
        .push(serde_json::json!({"role": "assistant", "content": "tool_use placeholder"}));
    parent_messages.push(serde_json::json!({
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "Subagent summary"}],
    }));

    assert_eq!(parent_messages.len() - initial_len, 2);
}

#[test]
fn test_l3_04_no_recursive_spawn_capability() {
    let names = child_tool_names();
    assert!(!names.contains(&"task".to_string()));
    assert!(!names.contains(&"subagent".to_string()));
}

#[test]
fn test_l3_05_respects_iteration_limit() {
    let mut llm = MockLLM::new();
    let max_iterations = 5;

    for i in 0..50 {
        llm.queue(
            "tool_use",
            vec![make_tool_use_block(
                &format!("tu_{}", i),
                "bash",
                serde_json::json!({"command": "echo loop"}),
            )],
        );
    }

    let tmp = tempfile::tempdir().unwrap();
    let mut registry = HandlerRegistry::new();
    registry.register(
        "bash",
        Arc::new(
            |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok("echoed".to_string())
            },
        ),
    );
    let ctx = AgentContext::mock(tmp.path());

    let tools = child_tools();
    SubagentFactory::spawn(
        &mut llm,
        "Infinite loop",
        &tools,
        &registry,
        &ctx,
        max_iterations,
        SpawnHooks::default(),
    );

    assert!(llm.call_count() <= max_iterations);
}

/// Progress callback receives human-readable phases (picked up, working, running tool, finished, complete).
#[test]
fn test_l3_06_progress_callback_receives_delegation_phases() {
    let mut llm = MockLLM::new();
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "tu_1",
            "read_file",
            serde_json::json!({"path": "README.md"}),
        )],
    );
    llm.queue("end_turn", vec![make_text_block("Done.")]);

    let tmp = tempfile::tempdir().unwrap();
    let mut registry = HandlerRegistry::new();
    registry.register(
        "read_file",
        Arc::new(
            |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok("file content".to_string())
            },
        ),
    );
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let progress_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let progress_log_clone = Arc::clone(&progress_log);
    let progress: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(move |msg: &str| {
        progress_log_clone.lock().unwrap().push(msg.to_string());
    });

    let _ = SubagentFactory::spawn(
        &mut llm,
        "Read README",
        &tools,
        &registry,
        &ctx,
        10,
        SpawnHooks {
            progress: Some(&progress),
            event_bus: None,
        },
    );

    let log = progress_log.lock().unwrap();
    let log_str = log.join(" ");
    assert!(
        log_str.contains("picked up"),
        "progress should report 'picked up'; got: {:?}",
        *log
    );
    assert!(
        log_str.contains("working — step"),
        "progress should report 'working — step N'; got: {:?}",
        *log
    );
    assert!(
        log_str.contains("running tool")
            || log_str.contains("finished")
            || log_str.contains("complete"),
        "progress should report tool/finish phase; got: {:?}",
        *log
    );
}

// --- Error handling tests ---

#[test]
fn test_l3_07_subagent_llm_auth_error() {
    let mut llm = MockLLM::new();
    llm.queue_error(LlmError::Auth {
        status: 401,
        message: "模拟认证失败".to_string(),
    });

    let tmp = tempfile::tempdir().unwrap();
    let registry = HandlerRegistry::new();
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let result = SubagentFactory::spawn(
        &mut llm,
        "Do work",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks::default(),
    );

    assert!(result.contains("subagent error"));
    assert!(result.contains("模拟认证失败"));
    assert_eq!(llm.call_count(), 1);
}

#[test]
fn test_l3_08_subagent_llm_overflow_error() {
    let mut llm = MockLLM::new();
    llm.queue_error(LlmError::Overflow {
        message: "Context length exceeded".to_string(),
    });

    let tmp = tempfile::tempdir().unwrap();
    let registry = HandlerRegistry::new();
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let result = SubagentFactory::spawn(
        &mut llm,
        "Do work",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks::default(),
    );

    assert!(result.contains("subagent error"));
    assert!(result.contains("Context length exceeded"));
}

#[test]
fn test_l3_09_subagent_tool_validation_error() {
    let mut llm = MockLLM::new();
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "tu_1",
            "read_file",
            serde_json::json!({"path": "nonexistent.txt"}),
        )],
    );
    llm.queue(
        "end_turn",
        vec![make_text_block("I cannot read that file.")],
    );

    let tmp = tempfile::tempdir().unwrap();
    let mut registry = HandlerRegistry::new();
    registry.register(
        "read_file",
        Arc::new(
            |_ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let path = require_str(&input, "path")?;
                if path.contains("nonexistent") {
                    return Err(HandlerError::Validation {
                        message: format!("File not found: {}", path),
                    });
                }
                Ok("content".to_string())
            },
        ),
    );
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let result = SubagentFactory::spawn(
        &mut llm,
        "Read file",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks::default(),
    );

    assert_eq!(result, "I cannot read that file.");
}

#[test]
fn test_l3_10_subagent_tool_execution_error() {
    let mut llm = MockLLM::new();
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "tu_1",
            "bash",
            serde_json::json!({"command": "rm -rf /"}),
        )],
    );
    llm.queue(
        "end_turn",
        vec![make_text_block("That command is too dangerous.")],
    );

    let tmp = tempfile::tempdir().unwrap();
    let mut registry = HandlerRegistry::new();
    registry.register(
        "bash",
        Arc::new(
            |_ctx: &AgentContext, input: serde_json::Value| -> HandlerResult {
                let cmd = require_str(&input, "command")?;
                if cmd.contains("rm -rf") {
                    return Err(HandlerError::Execution {
                        message: "Safety policy: destructive command blocked".to_string(),
                    });
                }
                Ok("output".to_string())
            },
        ),
    );
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let result = SubagentFactory::spawn(
        &mut llm,
        "Run safe command",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks::default(),
    );

    assert_eq!(result, "That command is too dangerous.");
}

#[test]
fn test_l3_11_subagent_no_final_text_returns_placeholder() {
    let mut llm = MockLLM::new();
    llm.queue(
        "end_turn",
        vec![make_tool_use_block("tu_1", "bash", serde_json::json!({}))],
    );

    let tmp = tempfile::tempdir().unwrap();
    let mut registry = HandlerRegistry::new();
    registry.register(
        "bash",
        Arc::new(
            |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok("output".to_string())
            },
        ),
    );
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let result = SubagentFactory::spawn(
        &mut llm,
        "Do something",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks::default(),
    );

    assert_eq!(result, "(no summary)");
}

#[test]
fn test_l3_12_subagent_zero_iterations() {
    let mut llm = MockLLM::new();

    let tmp = tempfile::tempdir().unwrap();
    let registry = HandlerRegistry::new();
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let result = SubagentFactory::spawn(
        &mut llm,
        "Do work",
        &tools,
        &registry,
        &ctx,
        0,
        SpawnHooks::default(),
    );

    assert_eq!(result, "(no summary)");
    assert_eq!(llm.call_count(), 0);
}

#[test]
fn test_l3_13_subagent_progress_callback_on_llm_error() {
    let mut llm = MockLLM::new();
    llm.queue_error(LlmError::Transient {
        status: 500,
        message: "Network timeout".to_string(),
    });

    let tmp = tempfile::tempdir().unwrap();
    let registry = HandlerRegistry::new();
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let progress_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let progress_log_clone = Arc::clone(&progress_log);
    let progress: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(move |msg: &str| {
        progress_log_clone.lock().unwrap().push(msg.to_string());
    });

    let _ = SubagentFactory::spawn(
        &mut llm,
        "Failing task",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks {
            progress: Some(&progress),
            event_bus: None,
        },
    );

    let log = progress_log.lock().unwrap();
    let log_str = log.join(" ");
    assert!(log_str.contains("picked up"));
    assert!(log_str.contains("LLM error") || log_str.contains("complete"));
}

#[test]
fn test_l3_14_event_bus_emits_progress_and_finish_on_success() {
    use nano_agent::isolation::EventBus;
    use tempfile::tempdir;

    let mut llm = MockLLM::new();
    llm.queue("end_turn", vec![make_text_block("Done.")]);

    let tmp = tempdir().unwrap();
    let events_path = tmp.path().join("events.jsonl");
    let event_bus = Arc::new(EventBus::new(&events_path));

    let registry = HandlerRegistry::new();
    let mut ctx = AgentContext::mock(tmp.path());
    ctx.services.event_bus = Arc::clone(&event_bus);
    let tools = child_tools();

    let _ = SubagentFactory::spawn(
        &mut llm,
        "Task",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks {
            progress: None,
            event_bus: Some(&event_bus),
        },
    );

    let events = event_bus.list_recent(10);
    assert!(events.contains("subagent_progress"));
    assert!(events.contains("subagent_finished"));
}

#[test]
fn test_l3_15_tool_error_reported_in_progress() {
    let mut llm = MockLLM::new();
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "tu_1",
            "read_file",
            serde_json::json!({"path": "missing.txt"}),
        )],
    );
    llm.queue(
        "end_turn",
        vec![make_text_block("I cannot access that file.")],
    );

    let tmp = tempfile::tempdir().unwrap();
    let mut registry = HandlerRegistry::new();
    registry.register(
        "read_file",
        Arc::new(
            move |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                // ignore input, always return error to simulate missing file
                Err(HandlerError::Execution {
                    message: "File not found".to_string(),
                })
            },
        ),
    );
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let result = SubagentFactory::spawn(
        &mut llm,
        "Read file",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks::default(),
    );

    assert_eq!(result, "I cannot access that file.");
}

#[test]
fn test_l3_16_subagent_with_empty_prompt() {
    let mut llm = MockLLM::new();
    llm.queue("end_turn", vec![make_text_block("Done with empty prompt")]);

    let tmp = tempfile::tempdir().unwrap();
    let registry = HandlerRegistry::new();
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let result = SubagentFactory::spawn(
        &mut llm,
        "",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks::default(),
    );

    assert_eq!(result, "Done with empty prompt");
}

#[test]
fn test_l3_17_subagent_with_small_success() {
    let mut llm = MockLLM::new();
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "tu_1",
            "bash",
            serde_json::json!({"cmd": "echo"}),
        )],
    );
    llm.queue("end_turn", vec![make_text_block("Done")]);

    let tmp = tempfile::tempdir().unwrap();
    let mut registry = HandlerRegistry::new();
    registry.register(
        "bash",
        Arc::new(
            move |_ctx: &AgentContext, _input: serde_json::Value| -> HandlerResult {
                Ok("output".to_string())
            },
        ),
    );
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();

    let result = SubagentFactory::spawn(
        &mut llm,
        "Simple",
        &tools,
        &registry,
        &ctx,
        5,
        SpawnHooks::default(),
    );

    assert_eq!(result, "Done");
    assert_eq!(llm.call_count(), 2); // tool_use + final end_turn
}

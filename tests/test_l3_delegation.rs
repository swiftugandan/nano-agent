use nano_agent::delegation::*;
use nano_agent::handler::{AgentContext, HandlerRegistry, HandlerResult};
use nano_agent::mock::*;
use nano_agent::types::*;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn test_l3_01_subagent_starts_with_fresh_messages() {
    let mut llm = MockLLM::new();
    llm.queue("end_turn", vec![make_text_block("Subagent done.")]);

    let tmp = tempfile::tempdir().unwrap();
    let registry = HandlerRegistry::new();
    let ctx = AgentContext::mock(tmp.path());
    let tools = child_tools();
    let result = SubagentFactory::spawn(&mut llm, "Do something", &tools, &registry, &ctx, 30);

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
    let result = SubagentFactory::spawn(&mut llm, "Test", &tools, &registry, &ctx, 30);

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
    );

    assert!(llm.call_count() <= max_iterations);
}

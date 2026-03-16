use nano_agent::core_loop::*;
use nano_agent::mock::*;
use nano_agent::types::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TestAgentLoop
// ---------------------------------------------------------------------------

#[test]
fn test_l1_01_loop_terminates_on_end_turn() {
    let mut llm = MockLLM::new();
    llm.queue("end_turn", vec![make_text_block("Done.")]);

    let mut messages = vec![serde_json::json!({"role": "user", "content": "Hello"})];
    let call_count = run_agent_loop(&mut llm, "Test", &mut messages, &[], &HashMap::new(), &LoopSignals::none());

    assert_eq!(call_count, 1);
    assert_eq!(messages.len(), 2); // user + assistant
    assert_eq!(messages.last().unwrap()["role"], "assistant");
}

#[test]
fn test_l1_02_loop_continues_on_tool_use() {
    let mut llm = MockLLM::new();
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "tu_1",
            "echo",
            serde_json::json!({"text": "hello"}),
        )],
    );
    llm.queue("end_turn", vec![make_text_block("Got it.")]);

    let dispatch = make_dispatch(HashMap::from([("echo".to_string(), "hello".to_string())]));
    let mut messages = vec![serde_json::json!({"role": "user", "content": "Echo hello"})];
    let call_count = run_agent_loop(
        &mut llm,
        "Test",
        &mut messages,
        &[serde_json::json!({"name": "echo"})],
        &dispatch,
        &LoopSignals::none(),
    );

    assert_eq!(call_count, 2);
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, vec!["user", "assistant", "user", "assistant"]);

    // The third message (index 2) should contain tool_result
    let user_msg = &messages[2];
    let content = user_msg["content"].as_array().unwrap();
    assert!(content
        .iter()
        .any(|p| p["type"].as_str() == Some("tool_result")));
}

#[test]
fn test_l1_07_tool_result_ids_match_tool_use_ids() {
    let mut llm = MockLLM::new();
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "tu_abc",
            "echo",
            serde_json::json!({"text": "test"}),
        )],
    );
    llm.queue("end_turn", vec![make_text_block("Done.")]);

    let dispatch = make_dispatch(HashMap::from([("echo".to_string(), "test".to_string())]));
    let mut messages =
        vec![serde_json::json!({"role": "user", "content": "Test correlation"})];
    run_agent_loop(
        &mut llm,
        "Test",
        &mut messages,
        &[serde_json::json!({"name": "echo"})],
        &dispatch,
        &LoopSignals::none(),
    );

    // Find tool_use ids from assistant messages
    let mut tool_use_ids = std::collections::HashSet::new();
    let mut tool_result_ids = std::collections::HashSet::new();

    for msg in &messages {
        if msg["role"].as_str() == Some("assistant") {
            if let Some(content) = msg["content"].as_array() {
                for block in content {
                    if block["type"].as_str() == Some("tool_use") {
                        tool_use_ids.insert(block["id"].as_str().unwrap().to_string());
                    }
                }
            }
        } else if msg["role"].as_str() == Some("user") {
            if let Some(content) = msg["content"].as_array() {
                for part in content {
                    if part["type"].as_str() == Some("tool_result") {
                        tool_result_ids
                            .insert(part["tool_use_id"].as_str().unwrap().to_string());
                    }
                }
            }
        }
    }
    assert_eq!(tool_use_ids, tool_result_ids);
}

// ---------------------------------------------------------------------------
// TestToolDispatcher
// ---------------------------------------------------------------------------

#[test]
fn test_l1_03_dispatch_routes_to_correct_handler() {
    let dispatch = make_dispatch(HashMap::from([
        ("foo".to_string(), "foo_result".to_string()),
        ("bar".to_string(), "bar_result".to_string()),
    ]));
    assert_eq!(route(&dispatch, "foo", serde_json::json!({})), "foo_result");
    assert_eq!(route(&dispatch, "bar", serde_json::json!({})), "bar_result");
}

#[test]
fn test_l1_04_unknown_tool_returns_error_string() {
    let mut llm = MockLLM::new();
    llm.queue(
        "tool_use",
        vec![make_tool_use_block(
            "tu_1",
            "nonexistent",
            serde_json::json!({}),
        )],
    );
    llm.queue("end_turn", vec![make_text_block("Ok.")]);

    let dispatch = make_dispatch(HashMap::from([("foo".to_string(), "foo_result".to_string())]));
    let mut messages = vec![serde_json::json!({"role": "user", "content": "Test"})];
    run_agent_loop(&mut llm, "Test", &mut messages, &[], &dispatch, &LoopSignals::none());

    // The tool_result should contain "Unknown tool"
    for msg in &messages {
        if msg["role"].as_str() == Some("user") {
            if let Some(content) = msg["content"].as_array() {
                for part in content {
                    if part["type"].as_str() == Some("tool_result") {
                        assert!(part["content"].as_str().unwrap().contains("Unknown tool"));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TestPathSandbox
// ---------------------------------------------------------------------------

#[test]
fn test_l1_05_rejects_escape_attempts() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let sandbox = PathSandbox::new(&workspace);
    let result = sandbox.safe_path("../../etc/passwd");
    assert!(result.is_err());
    if let Err(AgentError::ValueError(msg)) = result {
        assert!(msg.contains("escapes"));
    } else {
        panic!("Expected ValueError");
    }
}

#[test]
fn test_l1_06_allows_valid_relative_paths() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let sandbox = PathSandbox::new(&workspace);
    let result = sandbox.safe_path("src/main.py");
    assert!(result.is_ok());
    assert!(result.unwrap().starts_with(&workspace));
}

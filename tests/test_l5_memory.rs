use nano_agent::memory::*;
use nano_agent::mock::*;
use nano_agent::types::*;

fn make_tool_exchange(tool_name: &str, tool_id: &str, result_content: &str) -> (serde_json::Value, serde_json::Value) {
    let assistant_msg = serde_json::json!({
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": {},
        }],
    });
    let user_msg = serde_json::json!({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": result_content,
        }],
    });
    (assistant_msg, user_msg)
}

fn make_messages_with_tool_results(count: usize) -> Vec<serde_json::Value> {
    let mut messages = vec![serde_json::json!({"role": "user", "content": "Start"})];
    for i in 0..count {
        let result_content = format!("Result content for tool_{} ", i).repeat(30);
        let (asst, user) = make_tool_exchange(
            &format!("tool_{}", i),
            &format!("id_{}", i),
            &result_content,
        );
        messages.push(asst);
        messages.push(user);
    }
    messages
}

// ---------------------------------------------------------------------------
// TestMicroCompactor
// ---------------------------------------------------------------------------

#[test]
fn test_l5_01_preserves_last_n_tool_results() {
    let mut messages = make_messages_with_tool_results(6);
    micro_compact(&mut messages);

    // Collect tool_results
    let mut tool_results: Vec<String> = Vec::new();
    for msg in &messages {
        if msg["role"].as_str() == Some("user") {
            if let Some(content) = msg["content"].as_array() {
                for part in content {
                    if part["type"].as_str() == Some("tool_result") {
                        tool_results.push(part["content"].as_str().unwrap_or("").to_string());
                    }
                }
            }
        }
    }

    // Last KEEP_RECENT should be preserved
    let preserved = &tool_results[tool_results.len() - KEEP_RECENT..];
    let older = &tool_results[..tool_results.len() - KEEP_RECENT];

    for tr in preserved {
        assert!(tr.contains("Result content for"));
    }
    for tr in older {
        assert!(tr.contains("[Previous:"));
    }
}

#[test]
fn test_l5_02_placeholders_not_empty() {
    let mut messages = make_messages_with_tool_results(5);
    micro_compact(&mut messages);

    let mut tool_results: Vec<String> = Vec::new();
    for msg in &messages {
        if msg["role"].as_str() == Some("user") {
            if let Some(content) = msg["content"].as_array() {
                for part in content {
                    if part["type"].as_str() == Some("tool_result") {
                        tool_results.push(part["content"].as_str().unwrap_or("").to_string());
                    }
                }
            }
        }
    }

    let placeholders: Vec<&String> = tool_results
        .iter()
        .filter(|tr| tr.contains("[Previous:"))
        .collect();
    assert!(!placeholders.is_empty());
    for p in placeholders {
        assert!(!p.is_empty());
        assert!(
            p.to_lowercase().contains("used") || p.to_lowercase().contains("previous")
        );
    }
}

// ---------------------------------------------------------------------------
// TestAutoCompactor
// ---------------------------------------------------------------------------

#[test]
fn test_l5_03_saves_transcript_before_replacing() {
    let dir = tempfile::tempdir().unwrap();
    let transcript_dir = dir.path().join("transcripts");
    std::fs::create_dir_all(&transcript_dir).unwrap();

    let messages = make_messages_with_tool_results(10);
    let mut llm = MockLLM::new();
    llm.queue(
        "end_turn",
        vec![ContentBlock::Text {
            text: "Summary of conversation.".to_string(),
        }],
    );

    let (_result, _path) = auto_compact(&messages, &mut llm, &transcript_dir);

    // Check transcript file was saved
    let transcript_files: Vec<_> = std::fs::read_dir(&transcript_dir)
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .map_or(false, |n| n.starts_with("transcript_"))
        })
        .collect();
    assert!(transcript_files.len() >= 1);
    let content = std::fs::read_to_string(transcript_files[0].path()).unwrap();
    assert!(!content.is_empty());
}

#[test]
fn test_l5_04_auto_compact_returns_exactly_2_messages() {
    let dir = tempfile::tempdir().unwrap();
    let transcript_dir = dir.path().join("transcripts");
    std::fs::create_dir_all(&transcript_dir).unwrap();

    let messages = make_messages_with_tool_results(10);
    let mut llm = MockLLM::new();
    llm.queue(
        "end_turn",
        vec![ContentBlock::Text {
            text: "Summary of conversation.".to_string(),
        }],
    );

    let (result, _path) = auto_compact(&messages, &mut llm, &transcript_dir);

    assert_eq!(result.len(), 2);
    assert_eq!(result[0]["role"].as_str().unwrap(), "user");
    assert_eq!(result[1]["role"].as_str().unwrap(), "assistant");
    let content = result[0]["content"].as_str().unwrap().to_lowercase();
    assert!(content.contains("compressed") || content.contains("conversation"));
}

#[test]
fn test_l5_05_triggers_at_token_threshold() {
    // Create messages that exceed threshold
    let big_content = "x".repeat(THRESHOLD * 4 + 1000);
    let messages = vec![serde_json::json!({"role": "user", "content": big_content})];
    assert!(estimate_tokens(&messages) > THRESHOLD);
}

#[test]
fn test_l5_06_manual_compact_same_format_as_auto() {
    let dir = tempfile::tempdir().unwrap();
    let transcript_dir = dir.path().join("transcripts");
    std::fs::create_dir_all(&transcript_dir).unwrap();

    let messages = make_messages_with_tool_results(5);
    let mut llm = MockLLM::new();
    llm.queue(
        "end_turn",
        vec![ContentBlock::Text {
            text: "Manual summary.".to_string(),
        }],
    );

    let (result, _path) = auto_compact(&messages, &mut llm, &transcript_dir);

    assert_eq!(result.len(), 2);
    assert_eq!(result[0]["role"].as_str().unwrap(), "user");
    assert_eq!(result[1]["role"].as_str().unwrap(), "assistant");
}

use nano_agent::openai::{
    translate_messages_outbound, translate_response_inbound, translate_tools_outbound,
};
use nano_agent::types::*;

#[test]
fn test_openai_outbound_simple_messages() {
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: MessageContent::Text("Hello".to_string()),
        },
        Message {
            role: "assistant".to_string(),
            content: MessageContent::Text("Hi there".to_string()),
        },
    ];

    let result = translate_messages_outbound("You are helpful.", &messages);

    assert_eq!(result.len(), 3); // system + user + assistant
    assert_eq!(result[0]["role"], "system");
    assert_eq!(result[0]["content"], "You are helpful.");
    assert_eq!(result[1]["role"], "user");
    assert_eq!(result[1]["content"], "Hello");
    assert_eq!(result[2]["role"], "assistant");
    assert_eq!(result[2]["content"], "Hi there");
}

#[test]
fn test_openai_outbound_tool_use_and_result() {
    let messages = vec![
        Message {
            role: "assistant".to_string(),
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Let me check.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ]),
        },
        Message {
            role: "user".to_string(),
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "file1.txt\nfile2.txt".to_string(),
            }]),
        },
    ];

    let result = translate_messages_outbound("", &messages);

    // No system message (empty string)
    // Assistant with tool_calls
    assert_eq!(result[0]["role"], "assistant");
    assert_eq!(result[0]["content"], "Let me check.");
    let tool_calls = result[0]["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["id"], "call_1");
    assert_eq!(tool_calls[0]["type"], "function");
    assert_eq!(tool_calls[0]["function"]["name"], "bash");

    // Tool result → role: "tool"
    assert_eq!(result[1]["role"], "tool");
    assert_eq!(result[1]["tool_call_id"], "call_1");
    assert_eq!(result[1]["content"], "file1.txt\nfile2.txt");
}

#[test]
fn test_openai_outbound_tools_format() {
    let tools = vec![serde_json::json!({
        "name": "bash",
        "description": "Run a command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"]
        }
    })];

    let result = translate_tools_outbound(&tools);

    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["type"], "function");
    assert_eq!(result[0]["function"]["name"], "bash");
    assert_eq!(result[0]["function"]["description"], "Run a command");
    assert_eq!(
        result[0]["function"]["parameters"]["properties"]["command"]["type"],
        "string"
    );
}

#[test]
fn test_openai_inbound_text_response() {
    let json = serde_json::json!({
        "choices": [{
            "message": {
                "content": "Hello!",
                "tool_calls": null
            },
            "finish_reason": "stop"
        }]
    });

    let resp = translate_response_inbound(json);

    assert_eq!(resp.stop_reason, "end_turn");
    assert_eq!(resp.content.len(), 1);
    match &resp.content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "Hello!"),
        other => panic!("Expected Text, got {:?}", other),
    }
}

#[test]
fn test_openai_inbound_tool_calls() {
    let json = serde_json::json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": "{\"command\":\"ls\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let resp = translate_response_inbound(json);

    assert_eq!(resp.stop_reason, "tool_use");
    assert_eq!(resp.content.len(), 1);
    match &resp.content[0] {
        ContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "bash");
            assert_eq!(input["command"], "ls");
        }
        other => panic!("Expected ToolUse, got {:?}", other),
    }
}

#[test]
fn test_anthropic_message_serde_roundtrip() {
    let msg = Message {
        role: "assistant".to_string(),
        content: MessageContent::Blocks(vec![
            ContentBlock::Text {
                text: "thinking...".to_string(),
            },
            ContentBlock::ToolUse {
                id: "tu_1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "foo.rs"}),
            },
        ]),
    };

    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["role"], "assistant");
    let content = json["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "tool_use");
    assert_eq!(content[1]["name"], "read_file");

    // Round-trip back
    let parsed: Message = serde_json::from_value(json).unwrap();
    match &parsed.content {
        MessageContent::Blocks(blocks) => assert_eq!(blocks.len(), 2),
        _ => panic!("Expected Blocks"),
    }
}

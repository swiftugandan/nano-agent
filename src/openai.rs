use crate::types::*;
use serde::Deserialize;
use std::env;

/// OpenAI-compatible API backend (works with OpenRouter, local servers, etc).
pub struct OpenAiLlm {
    api_key: String,
    base_url: String,
    pub model: String,
}

impl OpenAiLlm {
    pub fn from_env() -> Self {
        Self {
            api_key: env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set"),
            base_url: env::var("OPENROUTER_BASE_URL")
                .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string()),
            model: env::var("OPENROUTER_MODEL")
                .unwrap_or_else(|_| "openai/gpt-4o-mini".to_string()),
        }
    }
}

// -- Response deserialization structs --

#[derive(Debug, Deserialize)]
struct OaiResponse {
    choices: Vec<OaiChoice>,
}

#[derive(Debug, Deserialize)]
struct OaiChoice {
    message: OaiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OaiMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OaiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OaiToolCall {
    id: String,
    function: OaiFunction,
}

#[derive(Debug, Deserialize)]
struct OaiFunction {
    name: String,
    arguments: String,
}

// -- Format translation --

/// Convert Anthropic-format messages to OpenAI-format messages.
pub fn translate_messages_outbound(system: &str, messages: &[Message]) -> Vec<serde_json::Value> {
    let mut out = Vec::new();

    // System message
    if !system.is_empty() {
        out.push(serde_json::json!({
            "role": "system",
            "content": system,
        }));
    }

    for msg in messages {
        match &msg.content {
            MessageContent::Text(s) => {
                out.push(serde_json::json!({
                    "role": msg.role,
                    "content": s,
                }));
            }
            MessageContent::Blocks(blocks) => {
                if msg.role == "assistant" {
                    // Collect text and tool_calls separately
                    let mut text_parts = Vec::new();
                    let mut tool_calls = Vec::new();

                    for block in blocks {
                        match block {
                            ContentBlock::Text { text } => {
                                text_parts.push(text.clone());
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                tool_calls.push(serde_json::json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": serde_json::to_string(input).unwrap(),
                                    }
                                }));
                            }
                            _ => {}
                        }
                    }

                    let mut msg_json = serde_json::json!({ "role": "assistant" });
                    if !text_parts.is_empty() {
                        msg_json["content"] = serde_json::Value::String(text_parts.join("\n"));
                    } else {
                        msg_json["content"] = serde_json::Value::Null;
                    }
                    if !tool_calls.is_empty() {
                        msg_json["tool_calls"] = serde_json::Value::Array(tool_calls);
                    }
                    out.push(msg_json);
                } else if msg.role == "user" {
                    // User blocks may contain tool_result blocks
                    for block in blocks {
                        match block {
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                            } => {
                                out.push(serde_json::json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": content,
                                }));
                            }
                            ContentBlock::Text { text } => {
                                out.push(serde_json::json!({
                                    "role": "user",
                                    "content": text,
                                }));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    out
}

/// Convert Anthropic tool definitions to OpenAI format.
pub fn translate_tools_outbound(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description").unwrap_or(&serde_json::Value::Null),
                    "parameters": t.get("input_schema").unwrap_or(&serde_json::Value::Null),
                }
            })
        })
        .collect()
}

/// Parse an OpenAI response into our Anthropic-style types.
pub fn translate_response_inbound(json: serde_json::Value) -> LlmResponse {
    let oai: OaiResponse = serde_json::from_value(json).expect("Failed to parse OpenAI response");

    let choice = &oai.choices[0];
    let mut content = Vec::new();

    if let Some(text) = &choice.message.content {
        if !text.is_empty() {
            content.push(ContentBlock::Text { text: text.clone() });
        }
    }

    if let Some(tool_calls) = &choice.message.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
            content.push(ContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
            });
        }
    }

    let stop_reason = match choice.finish_reason.as_deref() {
        Some("tool_calls") => "tool_use".to_string(),
        Some("stop") => "end_turn".to_string(),
        Some(other) => other.to_string(),
        None => "end_turn".to_string(),
    };

    LlmResponse {
        content,
        stop_reason,
    }
}

impl Llm for OpenAiLlm {
    fn create(&mut self, params: LlmParams) -> Result<LlmResponse, LlmError> {
        let oai_messages = translate_messages_outbound(&params.system, &params.messages);
        let oai_tools = translate_tools_outbound(&params.tools);

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": params.max_tokens,
        });

        if !oai_tools.is_empty() {
            body["tools"] = serde_json::Value::Array(oai_tools);
        }

        let url = format!("{}/chat/completions", self.base_url);
        let resp = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("content-type", "application/json")
            .send_json(body);

        match resp {
            Ok(response) => {
                let json: serde_json::Value = response
                    .into_json()
                    .expect("Failed to parse OpenAI response as JSON");
                Ok(translate_response_inbound(json))
            }
            Err(ureq::Error::Status(code, response)) => {
                let body = response.into_string().unwrap_or_default();
                Err(crate::resilience::classify_error(code, &body))
            }
            Err(e) => Err(LlmError::Fatal {
                message: e.to_string(),
            }),
        }
    }
}

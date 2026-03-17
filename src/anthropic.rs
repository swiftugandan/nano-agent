use crate::types::*;
use std::env;

/// Anthropic API backend.
pub struct AnthropicLlm {
    api_key: String,
    base_url: String,
    pub model: String,
}

impl AnthropicLlm {
    pub fn from_env() -> Self {
        Self {
            api_key: env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set"),
            base_url: env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
            model: env::var("MODEL_ID").unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string()),
        }
    }
}

impl Llm for AnthropicLlm {
    fn create(&mut self, params: LlmParams) -> Result<LlmResponse, LlmError> {
        let messages_json: Vec<serde_json::Value> = params
            .messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "system": params.system,
            "messages": messages_json,
            "tools": params.tools,
            "max_tokens": params.max_tokens,
        });

        let url = format!("{}/v1/messages", self.base_url);
        let resp = ureq::post(&url)
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", "2023-06-01")
            .set("content-type", "application/json")
            .send_json(body);

        match resp {
            Ok(response) => {
                let json: serde_json::Value = response
                    .into_json()
                    .expect("Failed to parse Anthropic response as JSON");

                let content: Vec<ContentBlock> = serde_json::from_value(json["content"].clone())
                    .expect("Failed to parse Anthropic content blocks");

                let stop_reason = json["stop_reason"]
                    .as_str()
                    .unwrap_or("end_turn")
                    .to_string();

                Ok(LlmResponse {
                    content,
                    stop_reason,
                })
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

use serde::{Deserialize, Serialize};

/// A minimal JSON-RPC 2.0 request envelope.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// A minimal JSON-RPC 2.0 success response.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcResponseOk<T: Serialize> {
    pub jsonrpc: &'static str,
    pub id: serde_json::Value,
    pub result: T,
}

/// A minimal JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcErrorObj {
    pub code: i64,
    pub message: String,
}

/// A minimal JSON-RPC 2.0 error response.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcResponseErr {
    pub jsonrpc: &'static str,
    pub id: serde_json::Value,
    pub error: JsonRpcErrorObj,
}

/// A server push notification (no `id`).
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcNotification<T: Serialize> {
    pub jsonrpc: &'static str,
    pub method: &'static str,
    pub params: T,
}

pub fn parse_request(text: &str) -> Result<JsonRpcRequest, String> {
    serde_json::from_str::<JsonRpcRequest>(text).map_err(|e| format!("Invalid JSON-RPC: {}", e))
}

pub fn ok<T: Serialize>(id: serde_json::Value, result: T) -> String {
    serde_json::to_string(&JsonRpcResponseOk {
        jsonrpc: "2.0",
        id,
        result,
    })
    .unwrap_or_else(|_| "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32603,\"message\":\"serialize error\"}}".to_string())
}

pub fn err(id: serde_json::Value, code: i64, message: impl Into<String>) -> String {
    serde_json::to_string(&JsonRpcResponseErr {
        jsonrpc: "2.0",
        id,
        error: JsonRpcErrorObj {
            code,
            message: message.into(),
        },
    })
    .unwrap_or_else(|_| "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32603,\"message\":\"serialize error\"}}".to_string())
}

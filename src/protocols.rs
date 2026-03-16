use crate::teams::MessageBus;
use std::collections::HashMap;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Request trackers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ShutdownRequest {
    pub target: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct PlanRequest {
    pub from: String,
    pub plan: String,
    pub status: String,
}

pub struct RequestTracker {
    pub shutdown_requests: Mutex<HashMap<String, ShutdownRequest>>,
    pub plan_requests: Mutex<HashMap<String, PlanRequest>>,
}

impl RequestTracker {
    pub fn new() -> Self {
        Self {
            shutdown_requests: Mutex::new(HashMap::new()),
            plan_requests: Mutex::new(HashMap::new()),
        }
    }

    pub fn clear(&self) {
        self.shutdown_requests.lock().unwrap().clear();
        self.plan_requests.lock().unwrap().clear();
    }

    /// Send a shutdown request to a teammate. Returns status string.
    pub fn handle_shutdown_request(&self, bus: &MessageBus, teammate: &str) -> String {
        let req_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        {
            let mut reqs = self.shutdown_requests.lock().unwrap();
            reqs.insert(
                req_id.clone(),
                ShutdownRequest {
                    target: teammate.to_string(),
                    status: "pending".to_string(),
                },
            );
        }
        bus.send_typed(
            "lead",
            teammate,
            "Please shut down gracefully.",
            "shutdown_request",
            Some(serde_json::json!({"request_id": req_id})),
        );
        format!(
            "Shutdown request {} sent to '{}' (status: pending)",
            req_id, teammate
        )
    }

    /// Handle a plan review (approve or reject).
    pub fn handle_plan_review(
        &self,
        bus: &MessageBus,
        request_id: &str,
        approve: bool,
        feedback: &str,
    ) -> String {
        let from_name;
        {
            let mut reqs = self.plan_requests.lock().unwrap();
            match reqs.get_mut(request_id) {
                Some(req) => {
                    req.status = if approve {
                        "approved".to_string()
                    } else {
                        "rejected".to_string()
                    };
                    from_name = req.from.clone();
                }
                None => {
                    return format!("Error: Unknown plan request_id '{}'", request_id);
                }
            }
        }

        bus.send_typed(
            "lead",
            &from_name,
            feedback,
            "plan_approval_response",
            Some(serde_json::json!({
                "request_id": request_id,
                "approve": approve,
                "feedback": feedback,
            })),
        );

        let status_str = if approve { "approved" } else { "rejected" };
        format!("Plan {} for '{}'", status_str, from_name)
    }
}

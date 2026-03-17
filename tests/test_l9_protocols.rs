use nano_agent::protocols::*;
use nano_agent::teams::MessageBus;

#[test]
fn test_l9_01_shutdown_request_creates_pending_entry() {
    let dir = tempfile::tempdir().unwrap();
    let inbox_dir = dir.path().join("inbox");
    std::fs::create_dir_all(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let tracker = RequestTracker::new();

    let result = tracker.handle_shutdown_request(&bus, "worker1");
    assert!(result.contains("pending"));

    // Check tracker
    let reqs = tracker.shutdown_requests.lock().unwrap();
    assert_eq!(reqs.len(), 1);
    let req_id = reqs.keys().next().unwrap().clone();
    assert_eq!(reqs[&req_id].status, "pending");
    drop(reqs);

    // Check inbox has message
    let msgs = bus.read_inbox("worker1");
    assert!(msgs.len() >= 1);
    assert!(msgs
        .iter()
        .any(|m| m["type"].as_str() == Some("shutdown_request")));
}

#[test]
fn test_l9_02_approval_transitions_to_approved() {
    let dir = tempfile::tempdir().unwrap();
    let inbox_dir = dir.path().join("inbox");
    std::fs::create_dir_all(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let tracker = RequestTracker::new();

    tracker.handle_shutdown_request(&bus, "worker1");

    let req_id = {
        let reqs = tracker.shutdown_requests.lock().unwrap();
        reqs.keys().next().unwrap().clone()
    };

    // Simulate teammate approving
    {
        let mut reqs = tracker.shutdown_requests.lock().unwrap();
        reqs.get_mut(&req_id).unwrap().status = "approved".to_string();
    }

    let reqs = tracker.shutdown_requests.lock().unwrap();
    assert_eq!(reqs[&req_id].status, "approved");
}

#[test]
fn test_l9_03_rejection_transitions_to_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let inbox_dir = dir.path().join("inbox");
    std::fs::create_dir_all(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let tracker = RequestTracker::new();

    tracker.handle_shutdown_request(&bus, "worker1");

    let req_id = {
        let reqs = tracker.shutdown_requests.lock().unwrap();
        reqs.keys().next().unwrap().clone()
    };

    {
        let mut reqs = tracker.shutdown_requests.lock().unwrap();
        reqs.get_mut(&req_id).unwrap().status = "rejected".to_string();
    }

    let reqs = tracker.shutdown_requests.lock().unwrap();
    assert_eq!(reqs[&req_id].status, "rejected");
}

#[test]
fn test_l9_04_plan_approval_sends_response() {
    let dir = tempfile::tempdir().unwrap();
    let inbox_dir = dir.path().join("inbox");
    std::fs::create_dir_all(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let tracker = RequestTracker::new();

    // Simulate a plan request from a teammate
    let req_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
    {
        let mut reqs = tracker.plan_requests.lock().unwrap();
        reqs.insert(
            req_id.clone(),
            PlanRequest {
                from: "worker1".to_string(),
                plan: "Refactor auth module".to_string(),
                status: "pending".to_string(),
            },
        );
    }

    // Lead approves
    let result = tracker.handle_plan_review(&bus, &req_id, true, "Looks good");
    assert!(result.contains("approved"));

    // Check tracker
    let reqs = tracker.plan_requests.lock().unwrap();
    assert_eq!(reqs[&req_id].status, "approved");
    drop(reqs);

    // Check worker1's inbox has response
    let msgs = bus.read_inbox("worker1");
    assert!(msgs.len() >= 1);
    assert!(msgs
        .iter()
        .any(|m| m["type"].as_str() == Some("plan_approval_response")));
}

#[test]
fn test_l9_05_correlation_by_request_id_concurrent() {
    let dir = tempfile::tempdir().unwrap();
    let inbox_dir = dir.path().join("inbox");
    std::fs::create_dir_all(&inbox_dir).unwrap();

    let bus = MessageBus::new(&inbox_dir);
    let tracker = RequestTracker::new();

    tracker.handle_shutdown_request(&bus, "worker1");
    let req1_id = {
        let reqs = tracker.shutdown_requests.lock().unwrap();
        reqs.keys().next().unwrap().clone()
    };

    tracker.handle_shutdown_request(&bus, "worker2");
    let req2_id = {
        let reqs = tracker.shutdown_requests.lock().unwrap();
        reqs.keys().find(|k| **k != req1_id).unwrap().clone()
    };

    // Approve only req1
    {
        let mut reqs = tracker.shutdown_requests.lock().unwrap();
        reqs.get_mut(&req1_id).unwrap().status = "approved".to_string();
    }

    let reqs = tracker.shutdown_requests.lock().unwrap();
    assert_eq!(reqs[&req1_id].status, "approved");
    assert_eq!(reqs[&req2_id].status, "pending");
}

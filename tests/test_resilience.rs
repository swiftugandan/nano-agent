use nano_agent::mock::*;
use nano_agent::resilience::*;
use nano_agent::types::*;

#[test]
fn test_retry_on_transient_error() {
    let mut mock = MockLLM::new();
    // Queue 2 transient errors then success
    mock.queue_error(LlmError::Transient {
        status: 429,
        message: "rate limited".into(),
    });
    mock.queue_error(LlmError::Transient {
        status: 503,
        message: "service unavailable".into(),
    });
    mock.queue("end_turn", vec![make_text_block("Success!")]);

    let policy = RetryPolicy {
        max_attempts: 5,
        base_delay_ms: 1, // fast for tests
        max_delay_ms: 10,
        jitter_factor: 0.0,
    };
    let auth = AuthProfile::empty();
    let mut resilient = ResilientLlm::new(Box::new(mock), policy, auth);

    let result = resilient.create(LlmParams {
        model: "test".into(),
        system: "test".into(),
        messages: vec![],
        tools: vec![],
        max_tokens: 100,
    });

    assert!(result.is_ok());
    let resp = result.unwrap();
    assert_eq!(resp.stop_reason, "end_turn");
}

#[test]
fn test_fatal_propagates_immediately() {
    let mut mock = MockLLM::new();
    mock.queue_error(LlmError::Fatal {
        message: "bad request".into(),
    });
    // This should never be reached
    mock.queue("end_turn", vec![make_text_block("Never")]);

    let policy = RetryPolicy {
        max_attempts: 5,
        base_delay_ms: 1,
        max_delay_ms: 10,
        jitter_factor: 0.0,
    };
    let mut resilient = ResilientLlm::new(Box::new(mock), policy, AuthProfile::empty());

    let result = resilient.create(LlmParams {
        model: "test".into(),
        system: "test".into(),
        messages: vec![],
        tools: vec![],
        max_tokens: 100,
    });

    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Fatal { message } => assert!(message.contains("bad request")),
        other => panic!("Expected Fatal, got {:?}", other),
    }
}

#[test]
fn test_overflow_propagates() {
    let mut mock = MockLLM::new();
    mock.queue_error(LlmError::Overflow {
        message: "context too long".into(),
    });

    let policy = RetryPolicy {
        max_attempts: 3,
        base_delay_ms: 1,
        max_delay_ms: 10,
        jitter_factor: 0.0,
    };
    let mut resilient = ResilientLlm::new(Box::new(mock), policy, AuthProfile::empty());

    let result = resilient.create(LlmParams {
        model: "test".into(),
        system: "test".into(),
        messages: vec![],
        tools: vec![],
        max_tokens: 100,
    });

    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Overflow { .. } => {}
        other => panic!("Expected Overflow, got {:?}", other),
    }
}

#[test]
fn test_max_attempts_exceeded() {
    let mut mock = MockLLM::new();
    // Queue more transient errors than max_attempts
    for _ in 0..5 {
        mock.queue_error(LlmError::Transient {
            status: 500,
            message: "internal error".into(),
        });
    }

    let policy = RetryPolicy {
        max_attempts: 3,
        base_delay_ms: 1,
        max_delay_ms: 10,
        jitter_factor: 0.0,
    };
    let mut resilient = ResilientLlm::new(Box::new(mock), policy, AuthProfile::empty());

    let result = resilient.create(LlmParams {
        model: "test".into(),
        system: "test".into(),
        messages: vec![],
        tools: vec![],
        max_tokens: 100,
    });

    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Transient { status, .. } => assert_eq!(status, 500),
        other => panic!("Expected Transient, got {:?}", other),
    }
}

#[test]
fn test_classify_error() {
    // Transient status codes
    match classify_error(429, "rate limited") {
        LlmError::Transient { status, .. } => assert_eq!(status, 429),
        other => panic!("Expected Transient for 429, got {:?}", other),
    }
    match classify_error(500, "internal") {
        LlmError::Transient { status, .. } => assert_eq!(status, 500),
        other => panic!("Expected Transient for 500, got {:?}", other),
    }
    match classify_error(502, "bad gateway") {
        LlmError::Transient { status, .. } => assert_eq!(status, 502),
        other => panic!("Expected Transient for 502, got {:?}", other),
    }
    match classify_error(503, "unavailable") {
        LlmError::Transient { status, .. } => assert_eq!(status, 503),
        other => panic!("Expected Transient for 503, got {:?}", other),
    }

    // Auth status codes
    match classify_error(401, "unauthorized") {
        LlmError::Auth { status, .. } => assert_eq!(status, 401),
        other => panic!("Expected Auth for 401, got {:?}", other),
    }
    match classify_error(403, "forbidden") {
        LlmError::Auth { status, .. } => assert_eq!(status, 403),
        other => panic!("Expected Auth for 403, got {:?}", other),
    }

    // Overflow detection by body keywords
    match classify_error(400, "context length exceeded") {
        LlmError::Overflow { .. } => {}
        other => panic!("Expected Overflow for context keyword, got {:?}", other),
    }
    match classify_error(400, "maximum token limit") {
        LlmError::Overflow { .. } => {}
        other => panic!("Expected Overflow for token keyword, got {:?}", other),
    }

    // Generic fatal
    match classify_error(400, "invalid json") {
        LlmError::Fatal { .. } => {}
        other => panic!("Expected Fatal for generic 400, got {:?}", other),
    }
}

#[test]
fn test_auth_rotation() {
    // We can't easily test key rotation with MockLLM because it doesn't use keys,
    // but we can verify the auth error path triggers rotation attempt
    let mut mock = MockLLM::new();
    mock.queue_error(LlmError::Auth {
        status: 401,
        message: "invalid key".into(),
    });
    // After rotation attempt (no extra keys), should propagate
    // No second response needed — the Auth error propagates immediately when rotation fails

    let policy = RetryPolicy {
        max_attempts: 3,
        base_delay_ms: 1,
        max_delay_ms: 10,
        jitter_factor: 0.0,
    };
    let mut resilient = ResilientLlm::new(Box::new(mock), policy, AuthProfile::empty());

    let result = resilient.create(LlmParams {
        model: "test".into(),
        system: "test".into(),
        messages: vec![],
        tools: vec![],
        max_tokens: 100,
    });

    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Auth { status, .. } => assert_eq!(status, 401),
        other => panic!("Expected Auth, got {:?}", other),
    }
}

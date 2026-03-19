use nano_agent::agent_core::{AgentCore, AgentCoreHandle, AgentEvent};
use nano_agent::context::{MemorySeed, Projector, SeedCollector};
use nano_agent::handler::AgentContext;
use nano_agent::mock::{make_text_block, MockLLM};
use std::sync::{Arc, Mutex};

fn tmp_dir(prefix: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let unique = format!(
        "{}-{}",
        prefix,
        uuid::Uuid::new_v4().to_string().replace('-', "")
    );
    p.push(unique);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn recv_event(handle: &AgentCoreHandle) -> nano_agent::agent_core::AgentEventEnvelope {
    let rx = handle.event_rx.lock().unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(2))
        .expect("timed out waiting for agent event")
}

#[test]
fn agent_core_emits_turn_finished() {
    let ws = tmp_dir("nano-agent-core-test");
    let ctx = AgentContext::mock(&ws);

    let prompts_dir = ws.join("prompts");
    let transcript_dir = ws.join("transcripts");
    std::fs::create_dir_all(&prompts_dir).unwrap();
    std::fs::create_dir_all(&transcript_dir).unwrap();

    let projector = Arc::new(Projector::new(&ws.join("projections"), 8_000));
    let memory_seed = Arc::new(MemorySeed::new(Arc::clone(&ctx.services.memory_store)));
    let seed_collector = Arc::new(Mutex::new(SeedCollector::new()));
    seed_collector
        .lock()
        .unwrap()
        .register(Arc::clone(&memory_seed) as Arc<dyn nano_agent::context::Seed>);

    let mut llm = MockLLM::new();
    llm.queue("end_turn", vec![make_text_block("hello from core")]);

    let registry = Arc::new(nano_agent::handler::HandlerRegistry::new());
    let tool_defs = Arc::new(Vec::<serde_json::Value>::new());

    let handle = AgentCore::start(
        ctx,
        Box::new(llm),
        registry,
        tool_defs,
        "test-model".to_string(),
        prompts_dir,
        transcript_dir,
        projector,
        memory_seed,
        seed_collector,
    );

    AgentCore::enqueue_turn(
        &handle,
        "req-1".to_string(),
        None,
        "hi".to_string(),
        false,
    )
    .unwrap();

    // Expect TurnStarted then TurnFinished (no tools in between).
    let e1 = recv_event(&handle);
    match e1.event {
        AgentEvent::TurnStarted { request_id } => assert_eq!(request_id, "req-1"),
        other => panic!("unexpected first event: {:?}", other),
    }

    // Drain until TurnFinished (future additions may insert warnings).
    for _ in 0..10 {
        let env = recv_event(&handle);
        if let AgentEvent::TurnFinished {
            request_id,
            assistant_text,
        } = env.event
        {
            assert_eq!(request_id, "req-1");
            assert!(assistant_text.contains("hello from core"));
            return;
        }
    }
    panic!("did not observe TurnFinished");
}


use nano_agent::ws_jsonrpc;

#[test]
fn parse_jsonrpc_request() {
    let text = r#"{"jsonrpc":"2.0","id":"1","method":"agent.run_turn","params":{"prompt":"hi"}}"#;
    let req = ws_jsonrpc::parse_request(text).expect("should parse");
    assert_eq!(req.jsonrpc, "2.0");
    assert_eq!(req.method, "agent.run_turn");
    assert_eq!(req.id.as_str().unwrap(), "1");
    assert_eq!(req.params["prompt"].as_str().unwrap(), "hi");
}

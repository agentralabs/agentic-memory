//! Phase 6: Stress tests for context capture and temporal chaining.
//!
//! Tests: conversation_log, temporal chains, scale (1K+ operations),
//! edge cases (unicode, empty, special chars), and auto-capture context.

mod common;

use serde_json::json;

use agentic_memory_mcp::tools::ToolRegistry;

use common::fixtures::create_test_session;

// ============================================================================
// Helper: extract text from tool result
// ============================================================================

fn result_text(result: &agentic_memory_mcp::types::ToolCallResult) -> String {
    match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text.clone(),
        _ => panic!("Expected text content"),
    }
}

fn result_json(result: &agentic_memory_mcp::types::ToolCallResult) -> serde_json::Value {
    serde_json::from_str(&result_text(result)).unwrap()
}

// ============================================================================
// 1. Context Capture — conversation_log
// ============================================================================

#[tokio::test]
async fn test_conversation_log_basic() {
    let session = create_test_session();
    let result = ToolRegistry::call(
        "conversation_log",
        Some(json!({
            "user_message": "How do I read a file?",
            "agent_response": "Use the Read tool"
        })),
        &session,
    )
    .await
    .unwrap();

    let parsed = result_json(&result);
    assert!(parsed["node_id"].as_u64().is_some());
    assert_eq!(
        parsed["message"],
        "Conversation logged and linked to temporal chain"
    );
}

#[tokio::test]
async fn test_conversation_log_user_only() {
    let session = create_test_session();
    let result = ToolRegistry::call(
        "conversation_log",
        Some(json!({"user_message": "Hello agent"})),
        &session,
    )
    .await
    .unwrap();

    let parsed = result_json(&result);
    assert!(parsed["node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_conversation_log_agent_only() {
    let session = create_test_session();
    let result = ToolRegistry::call(
        "conversation_log",
        Some(json!({"agent_response": "Here is the result"})),
        &session,
    )
    .await
    .unwrap();

    let parsed = result_json(&result);
    assert!(parsed["node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_conversation_log_with_topic() {
    let session = create_test_session();
    let result = ToolRegistry::call(
        "conversation_log",
        Some(json!({
            "user_message": "Fix the auth bug",
            "agent_response": "Found the root cause",
            "topic": "debugging"
        })),
        &session,
    )
    .await
    .unwrap();

    let parsed = result_json(&result);
    assert!(parsed["node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_conversation_log_empty_fails() {
    let session = create_test_session();
    let result = ToolRegistry::call("conversation_log", Some(json!({})), &session).await;
    assert!(result.is_err());
}

// ============================================================================
// 2. Temporal Chain — consecutive operations linked
// ============================================================================

#[tokio::test]
async fn test_temporal_chain_consecutive_adds() {
    let session = create_test_session();

    // Add 5 nodes — each should be linked to the previous via TemporalNext
    let mut node_ids = Vec::new();
    for i in 0..5 {
        let result = ToolRegistry::call(
            "memory_add",
            Some(json!({
                "event_type": "fact",
                "content": format!("Fact number {i}")
            })),
            &session,
        )
        .await
        .unwrap();

        let parsed = result_json(&result);
        let id = parsed["node_id"].as_u64().unwrap();
        node_ids.push(id);

        // After the first node, edges_created should include temporal
        if i > 0 {
            assert!(
                parsed["edges_created"].as_u64().unwrap() >= 1,
                "Node {i} should have at least 1 edge (temporal)"
            );
        }
    }

    // All node IDs should be unique
    let unique: std::collections::HashSet<_> = node_ids.iter().collect();
    assert_eq!(unique.len(), 5, "All node IDs should be unique");
}

#[tokio::test]
async fn test_temporal_chain_conversation_log_links() {
    let session = create_test_session();

    // Log 3 conversations — each should chain to the previous
    let mut node_ids = Vec::new();
    for i in 0..3 {
        let result = ToolRegistry::call(
            "conversation_log",
            Some(json!({
                "user_message": format!("Question {i}"),
                "agent_response": format!("Answer {i}")
            })),
            &session,
        )
        .await
        .unwrap();

        let parsed = result_json(&result);
        node_ids.push(parsed["node_id"].as_u64().unwrap());

        // After the first, should have temporal edges
        if i > 0 {
            assert_eq!(parsed["edges_created"], 1, "Should have 1 temporal edge");
        }
    }

    assert_eq!(node_ids.len(), 3);
    // Verify IDs are monotonically increasing (new nodes appended)
    for w in node_ids.windows(2) {
        assert!(w[1] > w[0], "Node IDs should be increasing");
    }
}

#[tokio::test]
async fn test_temporal_chain_mixed_operations() {
    let session = create_test_session();

    // Mix memory_add and conversation_log — all should chain together
    let r1 = ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Starting fact"})),
        &session,
    )
    .await
    .unwrap();
    let id1 = result_json(&r1)["node_id"].as_u64().unwrap();

    let r2 = ToolRegistry::call(
        "conversation_log",
        Some(json!({"user_message": "Discuss starting fact"})),
        &session,
    )
    .await
    .unwrap();
    let id2 = result_json(&r2)["node_id"].as_u64().unwrap();
    assert_eq!(result_json(&r2)["edges_created"], 1);

    let r3 = ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "decision", "content": "Based on discussion"})),
        &session,
    )
    .await
    .unwrap();
    let id3 = result_json(&r3)["node_id"].as_u64().unwrap();
    assert!(result_json(&r3)["edges_created"].as_u64().unwrap() >= 1);

    assert!(id1 < id2 && id2 < id3);
}

#[tokio::test]
async fn test_temporal_chain_resets_on_new_session() {
    let session = create_test_session();

    // Start a session, add nodes, end it
    ToolRegistry::call("session_start", Some(json!({})), &session)
        .await
        .unwrap();

    for i in 0..3 {
        ToolRegistry::call(
            "memory_add",
            Some(json!({"event_type": "fact", "content": format!("Session 1 fact {i}")})),
            &session,
        )
        .await
        .unwrap();
    }

    ToolRegistry::call(
        "session_end",
        Some(json!({"summary": "End of session 1"})),
        &session,
    )
    .await
    .unwrap();

    // Start a new session and add more nodes
    ToolRegistry::call("session_start", Some(json!({})), &session)
        .await
        .unwrap();

    let result = ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Session 2 first fact"})),
        &session,
    )
    .await
    .unwrap();

    // The first node in session 2 should exist (chain restarted)
    let parsed = result_json(&result);
    assert!(parsed["node_id"].as_u64().is_some());
}

// ============================================================================
// 3. Scale Tests — many operations
// ============================================================================

#[tokio::test]
async fn test_scale_1000_memory_adds() {
    let session = create_test_session();
    let start = std::time::Instant::now();

    for i in 0..1000 {
        let result = ToolRegistry::call(
            "memory_add",
            Some(json!({
                "event_type": if i % 3 == 0 { "fact" } else if i % 3 == 1 { "decision" } else { "inference" },
                "content": format!("Scale test event number {i} with some padding content to simulate real data")
            })),
            &session,
        )
        .await
        .unwrap();

        let parsed = result_json(&result);
        assert!(parsed["node_id"].as_u64().is_some());
    }

    let elapsed = start.elapsed();
    // Should complete in reasonable time (< 30 seconds for 1000 ops)
    assert!(
        elapsed.as_secs() < 30,
        "1000 memory adds took {:?} — too slow",
        elapsed
    );

    // Verify stats
    let stats = ToolRegistry::call("memory_stats", Some(json!({})), &session)
        .await
        .unwrap();
    let stats_json = result_json(&stats);
    assert_eq!(stats_json["node_count"], 1000);
}

#[tokio::test]
async fn test_scale_500_conversation_logs() {
    let session = create_test_session();
    let start = std::time::Instant::now();

    for i in 0..500 {
        ToolRegistry::call(
            "conversation_log",
            Some(json!({
                "user_message": format!("User message {i}"),
                "agent_response": format!("Agent response {i}"),
                "topic": format!("topic-{}", i % 10)
            })),
            &session,
        )
        .await
        .unwrap();
    }

    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 30,
        "500 conversation logs took {:?} — too slow",
        elapsed
    );

    // All 500 should be stored
    let stats = ToolRegistry::call("memory_stats", Some(json!({})), &session)
        .await
        .unwrap();
    let stats_json = result_json(&stats);
    assert_eq!(stats_json["node_count"], 500);
}

// ============================================================================
// 4. Edge Cases — unicode, special chars, long strings
// ============================================================================

#[tokio::test]
async fn test_unicode_content() {
    let session = create_test_session();

    let result = ToolRegistry::call(
        "conversation_log",
        Some(json!({
            "user_message": "用户消息 — 中文测试",
            "agent_response": "エージェントの応答 — 日本語テスト",
            "topic": "국제화"
        })),
        &session,
    )
    .await
    .unwrap();

    let parsed = result_json(&result);
    assert!(parsed["node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_emoji_content() {
    let session = create_test_session();

    let result = ToolRegistry::call(
        "memory_add",
        Some(json!({
            "event_type": "fact",
            "content": "User loves 🦀 Rust and 🐍 Python. Deployed to 🌍 production."
        })),
        &session,
    )
    .await
    .unwrap();

    assert!(result_json(&result)["node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_very_long_content() {
    let session = create_test_session();

    // 10KB string
    let long_content = "x".repeat(10_000);
    let result = ToolRegistry::call(
        "memory_add",
        Some(json!({
            "event_type": "fact",
            "content": long_content
        })),
        &session,
    )
    .await
    .unwrap();

    assert!(result_json(&result)["node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_special_chars_content() {
    let session = create_test_session();

    let result = ToolRegistry::call(
        "conversation_log",
        Some(json!({
            "user_message": "What about \"quotes\" and 'apostrophes'?",
            "agent_response": "Also: \\ backslash, \n newline, \t tab, & ampersand"
        })),
        &session,
    )
    .await
    .unwrap();

    assert!(result_json(&result)["node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_empty_topic() {
    let session = create_test_session();

    let result = ToolRegistry::call(
        "conversation_log",
        Some(json!({
            "user_message": "Test with empty topic",
            "topic": ""
        })),
        &session,
    )
    .await
    .unwrap();

    assert!(result_json(&result)["node_id"].as_u64().is_some());
}

// ============================================================================
// 5. Session Lifecycle with Context
// ============================================================================

#[tokio::test]
async fn test_full_session_lifecycle_with_context() {
    let session = create_test_session();

    // Start session
    let start = ToolRegistry::call("session_start", Some(json!({})), &session)
        .await
        .unwrap();
    assert!(result_json(&start)["session_id"].as_u64().is_some());

    // Log conversation
    ToolRegistry::call(
        "conversation_log",
        Some(json!({
            "user_message": "Help me build an API",
            "agent_response": "I'll help you set up a REST API"
        })),
        &session,
    )
    .await
    .unwrap();

    // Add related facts
    ToolRegistry::call(
        "memory_add",
        Some(json!({
            "event_type": "decision",
            "content": "Use Axum framework for the REST API"
        })),
        &session,
    )
    .await
    .unwrap();

    // End session
    let end = ToolRegistry::call(
        "session_end",
        Some(json!({
            "create_episode": true,
            "summary": "Helped user plan REST API with Axum"
        })),
        &session,
    )
    .await
    .unwrap();

    let end_json = result_json(&end);
    assert!(end_json["episode_node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_multiple_sessions_with_context() {
    let session = create_test_session();

    // Run 3 complete sessions
    for s in 0..3 {
        ToolRegistry::call("session_start", Some(json!({})), &session)
            .await
            .unwrap();

        for i in 0..5 {
            ToolRegistry::call(
                "conversation_log",
                Some(json!({
                    "user_message": format!("Session {s} question {i}"),
                    "agent_response": format!("Session {s} answer {i}")
                })),
                &session,
            )
            .await
            .unwrap();
        }

        ToolRegistry::call(
            "session_end",
            Some(json!({
                "create_episode": true,
                "summary": format!("Session {s} completed")
            })),
            &session,
        )
        .await
        .unwrap();
    }

    // Verify all nodes exist
    let stats = ToolRegistry::call("memory_stats", Some(json!({})), &session)
        .await
        .unwrap();
    let stats_json = result_json(&stats);
    // 3 sessions × 5 conversation_log nodes + 3 episode nodes = 18
    assert!(stats_json["node_count"].as_u64().unwrap() >= 18);
}

// ============================================================================
// 6. Regression — tool list includes conversation_log
// ============================================================================

#[tokio::test]
async fn test_tool_list_includes_conversation_log() {
    let tools = ToolRegistry::list_tools();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(
        names.contains(&"conversation_log"),
        "ToolRegistry must include conversation_log"
    );
}

#[tokio::test]
async fn test_conversation_log_definition_schema() {
    let tools = ToolRegistry::list_tools();
    let conv = tools.iter().find(|t| t.name == "conversation_log").unwrap();
    assert!(conv.description.is_some());
    let schema = &conv.input_schema;
    let props = schema["properties"].as_object().unwrap();
    assert!(props.contains_key("user_message"));
    assert!(props.contains_key("agent_response"));
    assert!(props.contains_key("topic"));
}

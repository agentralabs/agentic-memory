//! Phase 8: compact facade tool routing tests.

mod common;

use serde_json::json;

use agentic_memory_mcp::tools::ToolRegistry;
use agentic_memory_mcp::types::ToolContent;

use common::fixtures::create_test_session;

fn parse_text_json(result: &agentic_memory_mcp::types::ToolCallResult) -> serde_json::Value {
    let text = match &result.content[0] {
        ToolContent::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    serde_json::from_str(text).expect("tool result should be valid JSON")
}

#[test]
fn test_compact_tool_list_has_expected_surface() {
    let tools = ToolRegistry::list_tools_compact();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tools.len(), 10);
    assert!(names.contains(&"memory_core"));
    assert!(names.contains(&"memory_grounding"));
    assert!(names.contains(&"memory_workspace"));
    assert!(names.contains(&"memory_session"));
    assert!(names.contains(&"memory_infinite"));
    assert!(names.contains(&"memory_prophetic"));
    assert!(names.contains(&"memory_collective"));
    assert!(names.contains(&"memory_resurrection"));
    assert!(names.contains(&"memory_metamemory"));
    assert!(names.contains(&"memory_transcendent"));
}

#[tokio::test]
async fn test_compact_core_add_routes_to_memory_add() {
    let session = create_test_session();
    let result = ToolRegistry::call(
        "memory_core",
        Some(json!({
            "operation": "add",
            "params": {
                "event_type": "fact",
                "content": "Compact facade add",
                "confidence": 0.9
            }
        })),
        &session,
    )
    .await
    .expect("memory_core add should route");

    let parsed = parse_text_json(&result);
    assert_eq!(parsed["event_type"], "fact");
    assert!(parsed["node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_compact_workspace_create_routes() {
    let session = create_test_session();
    let result = ToolRegistry::call(
        "memory_workspace",
        Some(json!({
            "operation": "create",
            "params": {
                "name": "compact-facade-workspace"
            }
        })),
        &session,
    )
    .await
    .expect("memory_workspace create should route");

    let parsed = parse_text_json(&result);
    assert_eq!(parsed["name"], "compact-facade-workspace");
    assert!(parsed["workspace_id"].as_str().is_some());
}

#[tokio::test]
async fn test_compact_infinite_routes_to_invention_tools() {
    let session = create_test_session();
    let result = ToolRegistry::call(
        "memory_infinite",
        Some(json!({
            "operation": "immortal_stats"
        })),
        &session,
    )
    .await
    .expect("memory_infinite immortal_stats should route");

    let parsed = parse_text_json(&result);
    assert!(parsed["total_nodes"].is_number());
    assert!(parsed["immortality_score"].is_number());
}

//! Tool: session_start — Begin a new interaction session with prior context.
//!
//! Enhanced to solve the bootstrap problem: when a new session starts, the
//! response includes context from the last session (episode summary + session
//! gap information) so the agent doesn't start completely blank.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use agentic_memory::{EventType, PatternParams, PatternSort};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StartParams {
    session_id: Option<u32>,
    metadata: Option<Value>,
}

/// Return the tool definition for session_start.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "session_start".to_string(),
        description: Some("Start a new interaction session".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "integer", "description": "Optional explicit session ID" },
                "metadata": { "type": "object", "description": "Optional session metadata" }
            }
        }),
    }
}

/// Execute the session_start tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: StartParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let mut session = session.lock().await;
    let session_id = session.start_session(params.session_id)?;

    // ── Retrieve last session context (bootstrap problem solver) ──────────
    //
    // Query for the most recent episode node from a prior session.
    // This gives the agent immediate context about what happened last time.

    let graph = session.graph();
    let query = session.query_engine();

    let episode_pattern = PatternParams {
        event_types: vec![EventType::Episode],
        min_confidence: None,
        max_confidence: None,
        session_ids: vec![],
        created_after: None,
        created_before: None,
        min_decay_score: None,
        max_results: 1,
        sort_by: PatternSort::MostRecent,
    };

    let last_episode = query
        .pattern(graph, episode_pattern)
        .ok()
        .and_then(|eps| eps.into_iter().next())
        .map(|ep| {
            json!({
                "session_id": ep.session_id,
                "summary": ep.content,
                "created_at": ep.created_at,
            })
        });

    // Detect session gap.
    let all_sessions = graph.session_index().session_ids();
    let prev_session = all_sessions
        .iter()
        .filter(|&&s| s < session_id)
        .max()
        .copied();

    let total_sessions = all_sessions.len();

    Ok(ToolCallResult::json(&json!({
        "session_id": session_id,
        "message": format!("Session {session_id} started"),
        "total_sessions": total_sessions,
        "previous_session": prev_session,
        "last_episode": last_episode,
    })))
}

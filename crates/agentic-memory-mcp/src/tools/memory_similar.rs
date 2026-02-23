//! Tool: memory_similar — Find semantically similar memories.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use agentic_memory::{EventType, SimilarityParams, TextSearchParams};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
struct SimilarParams {
    query_text: Option<String>,
    query_vec: Option<Vec<f32>>,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_min_similarity")]
    min_similarity: f32,
    #[serde(default)]
    event_types: Vec<String>,
}

fn default_top_k() -> usize {
    10
}

fn default_min_similarity() -> f32 {
    0.5
}

/// Return the tool definition for memory_similar.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_similar".to_string(),
        description: Some("Find semantically similar memories using vector similarity".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query_text": { "type": "string" },
                "query_vec": { "type": "array", "items": { "type": "number" } },
                "top_k": { "type": "integer", "default": 10 },
                "min_similarity": { "type": "number", "default": 0.5 },
                "event_types": { "type": "array", "items": { "type": "string" } }
            }
        }),
    }
}

/// Execute the memory_similar tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: SimilarParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let event_types: Vec<EventType> = params
        .event_types
        .iter()
        .filter_map(|name| EventType::from_name(name))
        .collect();

    let session = session.lock().await;

    if let Some(query_vec) = params.query_vec {
        let similarity_params = SimilarityParams {
            query_vec,
            top_k: params.top_k,
            min_similarity: params.min_similarity,
            event_types,
            skip_zero_vectors: true,
        };

        let results = session
            .query_engine()
            .similarity(session.graph(), similarity_params)
            .map_err(|e| McpError::AgenticMemory(format!("Similarity search failed: {e}")))?;

        let matches: Vec<Value> = results
            .iter()
            .filter_map(|m| {
                session.graph().get_node(m.node_id).map(|node| {
                    json!({
                        "node_id": m.node_id,
                        "similarity": m.similarity,
                        "event_type": node.event_type.name(),
                        "content": node.content,
                        "confidence": node.confidence,
                    })
                })
            })
            .collect();

        return Ok(ToolCallResult::json(&json!({
            "mode": "vector",
            "count": matches.len(),
            "matches": matches,
        })));
    }

    let query_text = params.query_text.ok_or_else(|| {
        McpError::InvalidParams("Either query_vec or query_text is required".to_string())
    })?;

    let text_results = session
        .query_engine()
        .text_search(
            session.graph(),
            None,
            None,
            TextSearchParams {
                query: query_text,
                max_results: params.top_k,
                event_types,
                session_ids: Vec::new(),
                min_score: 0.0,
            },
        )
        .map_err(|e| McpError::AgenticMemory(format!("Text similarity fallback failed: {e}")))?;

    let matches: Vec<Value> = text_results
        .iter()
        .filter_map(|m| {
            session.graph().get_node(m.node_id).map(|node| {
                json!({
                    "node_id": m.node_id,
                    "similarity": m.score,
                    "text_score": m.score,
                    "matched_terms": m.matched_terms,
                    "event_type": node.event_type.name(),
                    "content": node.content,
                    "confidence": node.confidence,
                })
            })
        })
        .collect();

    Ok(ToolCallResult::json(&json!({
        "mode": "text_fallback",
        "count": matches.len(),
        "matches": matches,
    })))
}

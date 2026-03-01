//! Compact MCP facade tools for low-token operation.
//!
//! These tools expose operation-based routing while preserving the existing
//! fine-grained tool surface for backward compatibility.

use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

use super::{
    conversation_log, invention_collective, invention_infinite, invention_metamemory,
    invention_prophetic, invention_resurrection, invention_transcendent, memory_add, memory_causal,
    memory_context, memory_correct, memory_evidence, memory_ground, memory_quality, memory_query,
    memory_resolve, memory_session_resume, memory_similar, memory_stats, memory_suggest,
    memory_temporal, memory_traverse, memory_workspace_add, memory_workspace_compare,
    memory_workspace_create, memory_workspace_list, memory_workspace_query, memory_workspace_xref,
    session_end, session_start,
};

fn op_schema(ops: &[String], description: &str) -> Value {
    json!({
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ops,
                "description": description
            },
            "params": {
                "type": "object",
                "description": "Arguments for the selected operation"
            }
        }
    })
}

fn suffix_ops(defs: Vec<ToolDefinition>) -> Vec<String> {
    defs.into_iter()
        .filter_map(|d| d.name.strip_prefix("memory_").map(str::to_string))
        .collect()
}

pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "memory_core".to_string(),
            description: Some(
                "Compact core facade: add/query/quality/traverse/correct/resolve/context/similar/causal/temporal/stats/conversation_log".to_string(),
            ),
            input_schema: op_schema(
                &[
                    "conversation_log".to_string(),
                    "add".to_string(),
                    "query".to_string(),
                    "quality".to_string(),
                    "traverse".to_string(),
                    "correct".to_string(),
                    "resolve".to_string(),
                    "context".to_string(),
                    "similar".to_string(),
                    "causal".to_string(),
                    "temporal".to_string(),
                    "stats".to_string(),
                ],
                "Core memory operation",
            ),
        },
        ToolDefinition {
            name: "memory_grounding".to_string(),
            description: Some("Compact grounding facade: ground/evidence/suggest".to_string()),
            input_schema: op_schema(
                &[
                    "ground".to_string(),
                    "evidence".to_string(),
                    "suggest".to_string(),
                ],
                "Grounding operation",
            ),
        },
        ToolDefinition {
            name: "memory_workspace".to_string(),
            description: Some(
                "Compact workspace facade: create/add/list/query/compare/xref".to_string(),
            ),
            input_schema: op_schema(
                &[
                    "create".to_string(),
                    "add".to_string(),
                    "list".to_string(),
                    "query".to_string(),
                    "compare".to_string(),
                    "xref".to_string(),
                ],
                "Workspace operation",
            ),
        },
        ToolDefinition {
            name: "memory_session".to_string(),
            description: Some("Compact session facade: start/end/resume".to_string()),
            input_schema: op_schema(
                &["start".to_string(), "end".to_string(), "resume".to_string()],
                "Session operation",
            ),
        },
        ToolDefinition {
            name: "memory_infinite".to_string(),
            description: Some("Compact facade for INFINITE invention tools".to_string()),
            input_schema: op_schema(
                &suffix_ops(invention_infinite::all_definitions()),
                "INFINITE operation",
            ),
        },
        ToolDefinition {
            name: "memory_prophetic".to_string(),
            description: Some("Compact facade for PROPHETIC invention tools".to_string()),
            input_schema: op_schema(
                &suffix_ops(invention_prophetic::all_definitions()),
                "PROPHETIC operation",
            ),
        },
        ToolDefinition {
            name: "memory_collective".to_string(),
            description: Some("Compact facade for COLLECTIVE invention tools".to_string()),
            input_schema: op_schema(
                &suffix_ops(invention_collective::all_definitions()),
                "COLLECTIVE operation",
            ),
        },
        ToolDefinition {
            name: "memory_resurrection".to_string(),
            description: Some("Compact facade for RESURRECTION invention tools".to_string()),
            input_schema: op_schema(
                &suffix_ops(invention_resurrection::all_definitions()),
                "RESURRECTION operation",
            ),
        },
        ToolDefinition {
            name: "memory_metamemory".to_string(),
            description: Some("Compact facade for METAMEMORY invention tools".to_string()),
            input_schema: op_schema(
                &suffix_ops(invention_metamemory::all_definitions()),
                "METAMEMORY operation",
            ),
        },
        ToolDefinition {
            name: "memory_transcendent".to_string(),
            description: Some("Compact facade for TRANSCENDENT invention tools".to_string()),
            input_schema: op_schema(
                &suffix_ops(invention_transcendent::all_definitions()),
                "TRANSCENDENT operation",
            ),
        },
    ]
}

fn decode_operation(args: Value) -> McpResult<(String, Value)> {
    let obj = args
        .as_object()
        .ok_or_else(|| McpError::InvalidParams("arguments must be an object".to_string()))?;
    let operation = obj
        .get("operation")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidParams("'operation' is required".to_string()))?
        .to_string();

    if let Some(params) = obj.get("params") {
        return Ok((operation, params.clone()));
    }

    let mut passthrough = obj.clone();
    passthrough.remove("operation");
    Ok((operation, Value::Object(passthrough)))
}

fn with_memory_prefix(op: &str) -> String {
    if op.starts_with("memory_") {
        op.to_string()
    } else {
        format!("memory_{op}")
    }
}

pub async fn try_execute(
    name: &str,
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> Option<McpResult<ToolCallResult>> {
    if !matches!(
        name,
        "memory_core"
            | "memory_grounding"
            | "memory_workspace"
            | "memory_session"
            | "memory_infinite"
            | "memory_prophetic"
            | "memory_collective"
            | "memory_resurrection"
            | "memory_metamemory"
            | "memory_transcendent"
    ) {
        return None;
    }

    let (operation, params) = match decode_operation(args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };

    match name {
        "memory_core" => Some(match operation.as_str() {
            "conversation_log" => conversation_log::execute(params, session).await,
            "add" => memory_add::execute(params, session).await,
            "query" => memory_query::execute(params, session).await,
            "quality" => memory_quality::execute(params, session).await,
            "traverse" => memory_traverse::execute(params, session).await,
            "correct" => memory_correct::execute(params, session).await,
            "resolve" => memory_resolve::execute(params, session).await,
            "context" => memory_context::execute(params, session).await,
            "similar" => memory_similar::execute(params, session).await,
            "causal" => memory_causal::execute(params, session).await,
            "temporal" => memory_temporal::execute(params, session).await,
            "stats" => memory_stats::execute(params, session).await,
            _ => Err(McpError::InvalidParams(format!(
                "Unknown memory_core operation: {operation}"
            ))),
        }),
        "memory_grounding" => Some(match operation.as_str() {
            "ground" => memory_ground::execute(params, session).await,
            "evidence" => memory_evidence::execute(params, session).await,
            "suggest" => memory_suggest::execute(params, session).await,
            _ => Err(McpError::InvalidParams(format!(
                "Unknown memory_grounding operation: {operation}"
            ))),
        }),
        "memory_workspace" => Some(match operation.as_str() {
            "create" => memory_workspace_create::execute(params, session).await,
            "add" => memory_workspace_add::execute(params, session).await,
            "list" => memory_workspace_list::execute(params, session).await,
            "query" => memory_workspace_query::execute(params, session).await,
            "compare" => memory_workspace_compare::execute(params, session).await,
            "xref" => memory_workspace_xref::execute(params, session).await,
            _ => Err(McpError::InvalidParams(format!(
                "Unknown memory_workspace operation: {operation}"
            ))),
        }),
        "memory_session" => Some(match operation.as_str() {
            "start" => session_start::execute(params, session).await,
            "end" => session_end::execute(params, session).await,
            "resume" => memory_session_resume::execute(params, session).await,
            _ => Err(McpError::InvalidParams(format!(
                "Unknown memory_session operation: {operation}"
            ))),
        }),
        "memory_infinite" => Some({
            let routed = with_memory_prefix(&operation);
            if let Some(result) = invention_infinite::try_execute(&routed, params, session).await {
                result
            } else {
                Err(McpError::InvalidParams(format!(
                    "Unknown memory_infinite operation: {operation}"
                )))
            }
        }),
        "memory_prophetic" => Some({
            let routed = with_memory_prefix(&operation);
            if let Some(result) = invention_prophetic::try_execute(&routed, params, session).await {
                result
            } else {
                Err(McpError::InvalidParams(format!(
                    "Unknown memory_prophetic operation: {operation}"
                )))
            }
        }),
        "memory_collective" => Some({
            let routed = with_memory_prefix(&operation);
            if let Some(result) = invention_collective::try_execute(&routed, params, session).await
            {
                result
            } else {
                Err(McpError::InvalidParams(format!(
                    "Unknown memory_collective operation: {operation}"
                )))
            }
        }),
        "memory_resurrection" => Some({
            let routed = with_memory_prefix(&operation);
            if let Some(result) =
                invention_resurrection::try_execute(&routed, params, session).await
            {
                result
            } else {
                Err(McpError::InvalidParams(format!(
                    "Unknown memory_resurrection operation: {operation}"
                )))
            }
        }),
        "memory_metamemory" => Some({
            let routed = with_memory_prefix(&operation);
            if let Some(result) = invention_metamemory::try_execute(&routed, params, session).await
            {
                result
            } else {
                Err(McpError::InvalidParams(format!(
                    "Unknown memory_metamemory operation: {operation}"
                )))
            }
        }),
        "memory_transcendent" => Some({
            let routed = with_memory_prefix(&operation);
            if let Some(result) =
                invention_transcendent::try_execute(&routed, params, session).await
            {
                result
            } else {
                Err(McpError::InvalidParams(format!(
                    "Unknown memory_transcendent operation: {operation}"
                )))
            }
        }),
        _ => None,
    }
}

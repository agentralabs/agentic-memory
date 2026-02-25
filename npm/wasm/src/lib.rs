//! WebAssembly bindings for agentic-memory.
//!
//! Provides a JavaScript-friendly API wrapping the core MemoryGraph,
//! QueryEngine, and WriteEngine for use in Node.js and browser environments.

use wasm_bindgen::prelude::*;

use agentic_memory::{
    CognitiveEventBuilder, EventType, EdgeType, Edge,
    MemoryGraph, QueryEngine, WriteEngine,
    PatternParams, PatternSort, DEFAULT_DIMENSION,
};

use serde::Serialize;

/// Serializable node representation for JSON export.
#[derive(Serialize)]
struct NodeView {
    id: u64,
    event_type: String,
    content: String,
    confidence: f32,
    session_id: u32,
    created_at: u64,
    access_count: u32,
    decay_score: f32,
}

/// Serializable edge representation for JSON export.
#[derive(Serialize)]
struct EdgeView {
    source_id: u64,
    target_id: u64,
    edge_type: String,
    weight: f32,
    created_at: u64,
}

/// JavaScript-facing wrapper around the agentic-memory graph.
///
/// Usage from Node.js:
/// ```js
/// const { WasmMemoryGraph } = require('@agentic/memory');
/// const graph = new WasmMemoryGraph();
/// const id = graph.add_event("fact", "The sky is blue", 0.95);
/// console.log(graph.node_count());
/// ```
#[wasm_bindgen]
pub struct WasmMemoryGraph {
    graph: MemoryGraph,
    query_engine: QueryEngine,
    write_engine: WriteEngine,
}

#[wasm_bindgen]
impl WasmMemoryGraph {
    /// Create a new empty memory graph with the default dimension (128).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            graph: MemoryGraph::new(DEFAULT_DIMENSION),
            query_engine: QueryEngine::new(),
            write_engine: WriteEngine::new(DEFAULT_DIMENSION),
        }
    }

    /// Create a new empty memory graph with a custom feature vector dimension.
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            graph: MemoryGraph::new(dimension),
            query_engine: QueryEngine::new(),
            write_engine: WriteEngine::new(dimension),
        }
    }

    /// Add a cognitive event to the graph.
    ///
    /// `event_type` must be one of: "fact", "decision", "inference",
    /// "correction", "skill", "episode".
    ///
    /// Returns the assigned node ID.
    pub fn add_event(
        &mut self,
        event_type: &str,
        content: &str,
        confidence: f32,
    ) -> Result<u64, JsValue> {
        let et = EventType::from_name(event_type)
            .ok_or_else(|| JsValue::from_str(&format!("unknown event type: {}", event_type)))?;
        let event = CognitiveEventBuilder::new(et, content)
            .confidence(confidence)
            .build();
        self.graph
            .add_node(event)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Add a cognitive event with a specific session ID.
    ///
    /// Returns the assigned node ID.
    pub fn add_event_with_session(
        &mut self,
        event_type: &str,
        content: &str,
        confidence: f32,
        session_id: u32,
    ) -> Result<u64, JsValue> {
        let et = EventType::from_name(event_type)
            .ok_or_else(|| JsValue::from_str(&format!("unknown event type: {}", event_type)))?;
        let event = CognitiveEventBuilder::new(et, content)
            .confidence(confidence)
            .session_id(session_id)
            .build();
        self.graph
            .add_node(event)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Add an edge between two nodes.
    ///
    /// `edge_type` must be one of: "caused_by", "supports", "contradicts",
    /// "supersedes", "related_to", "part_of", "temporal_next".
    pub fn add_edge(
        &mut self,
        source_id: u64,
        target_id: u64,
        edge_type: &str,
        weight: f32,
    ) -> Result<(), JsValue> {
        let et = EdgeType::from_name(edge_type)
            .ok_or_else(|| JsValue::from_str(&format!("unknown edge type: {}", edge_type)))?;
        let edge = Edge::new(source_id, target_id, et, weight);
        self.graph
            .add_edge(edge)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Record a correction: marks old node as superseded, creates new node.
    ///
    /// Returns the new correction node ID.
    pub fn correct(
        &mut self,
        old_node_id: u64,
        new_content: &str,
        session_id: u32,
    ) -> Result<u64, JsValue> {
        self.write_engine
            .correct(&mut self.graph, old_node_id, new_content, session_id)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get the feature vector dimension.
    pub fn dimension(&self) -> usize {
        self.graph.dimension()
    }

    /// Get a single node as a JSON string.
    ///
    /// Returns null if the node does not exist.
    pub fn get_node(&self, id: u64) -> Result<JsValue, JsValue> {
        match self.graph.get_node(id) {
            Some(node) => {
                let view = NodeView {
                    id: node.id,
                    event_type: node.event_type.name().to_string(),
                    content: node.content.clone(),
                    confidence: node.confidence,
                    session_id: node.session_id,
                    created_at: node.created_at,
                    access_count: node.access_count,
                    decay_score: node.decay_score,
                };
                serde_json::to_string(&view)
                    .map(|s| JsValue::from_str(&s))
                    .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
            }
            None => Ok(JsValue::NULL),
        }
    }

    /// Query recent nodes, sorted by creation time (most recent first).
    ///
    /// Returns a JSON array string of nodes.
    pub fn query_recent(&self, limit: usize) -> Result<String, JsValue> {
        let params = PatternParams {
            event_types: vec![],
            min_confidence: None,
            max_confidence: None,
            session_ids: vec![],
            created_after: None,
            created_before: None,
            min_decay_score: None,
            max_results: limit,
            sort_by: PatternSort::MostRecent,
        };
        let results = self
            .query_engine
            .pattern(&self.graph, params)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
        let views: Vec<NodeView> = results
            .into_iter()
            .map(|n| NodeView {
                id: n.id,
                event_type: n.event_type.name().to_string(),
                content: n.content.clone(),
                confidence: n.confidence,
                session_id: n.session_id,
                created_at: n.created_at,
                access_count: n.access_count,
                decay_score: n.decay_score,
            })
            .collect();
        serde_json::to_string(&views)
            .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
    }

    /// Query nodes filtered by event type.
    ///
    /// Returns a JSON array string of matching nodes.
    pub fn query_by_type(&self, event_type: &str, limit: usize) -> Result<String, JsValue> {
        let et = EventType::from_name(event_type)
            .ok_or_else(|| JsValue::from_str(&format!("unknown event type: {}", event_type)))?;
        let params = PatternParams {
            event_types: vec![et],
            min_confidence: None,
            max_confidence: None,
            session_ids: vec![],
            created_after: None,
            created_before: None,
            min_decay_score: None,
            max_results: limit,
            sort_by: PatternSort::MostRecent,
        };
        let results = self
            .query_engine
            .pattern(&self.graph, params)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
        let views: Vec<NodeView> = results
            .into_iter()
            .map(|n| NodeView {
                id: n.id,
                event_type: n.event_type.name().to_string(),
                content: n.content.clone(),
                confidence: n.confidence,
                session_id: n.session_id,
                created_at: n.created_at,
                access_count: n.access_count,
                decay_score: n.decay_score,
            })
            .collect();
        serde_json::to_string(&views)
            .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
    }

    /// Remove a node and all its edges.
    ///
    /// Returns the removed node as a JSON string.
    pub fn remove_node(&mut self, id: u64) -> Result<String, JsValue> {
        let removed = self
            .graph
            .remove_node(id)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
        let view = NodeView {
            id: removed.id,
            event_type: removed.event_type.name().to_string(),
            content: removed.content.clone(),
            confidence: removed.confidence,
            session_id: removed.session_id,
            created_at: removed.created_at,
            access_count: removed.access_count,
            decay_score: removed.decay_score,
        };
        serde_json::to_string(&view)
            .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
    }

    /// Export the entire graph (all nodes and edges) as a JSON string.
    pub fn to_json(&self) -> Result<String, JsValue> {
        let nodes: Vec<NodeView> = self
            .graph
            .nodes()
            .iter()
            .map(|n| NodeView {
                id: n.id,
                event_type: n.event_type.name().to_string(),
                content: n.content.clone(),
                confidence: n.confidence,
                session_id: n.session_id,
                created_at: n.created_at,
                access_count: n.access_count,
                decay_score: n.decay_score,
            })
            .collect();
        let edges: Vec<EdgeView> = self
            .graph
            .edges()
            .iter()
            .map(|e| EdgeView {
                source_id: e.source_id,
                target_id: e.target_id,
                edge_type: e.edge_type.name().to_string(),
                weight: e.weight,
                created_at: e.created_at,
            })
            .collect();

        #[derive(Serialize)]
        struct GraphExport {
            nodes: Vec<NodeView>,
            edges: Vec<EdgeView>,
            node_count: usize,
            edge_count: usize,
        }

        let export = GraphExport {
            node_count: nodes.len(),
            edge_count: edges.len(),
            nodes,
            edges,
        };
        serde_json::to_string(&export)
            .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
    }
}

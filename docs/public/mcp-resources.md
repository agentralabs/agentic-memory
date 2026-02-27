---
status: stable
---

# MCP Resources

AgenticMemory exposes memory graph data through the `amem://` URI scheme in the MCP Resources API.

## URI Scheme

All resources use the `amem://` prefix.

## Concrete Resources

### `amem://graph/stats`

Return overall memory graph statistics.

**Format:** JSON object.

```json
{
  "node_count": 142,
  "edge_count": 215,
  "dimension": 128,
  "session_count": 8,
  "type_counts": {
    "fact": 85,
    "decision": 32,
    "inference": 15,
    "correction": 3,
    "skill": 2,
    "episode": 5
  }
}
```

### `amem://graph/recent`

Return the most recently created nodes (top 20).

**Format:** JSON object with count and node array.

```json
{
  "count": 20,
  "nodes": [
    {
      "id": 142,
      "event_type": "fact",
      "content": "The API uses JWT tokens for auth",
      "confidence": 0.95,
      "session_id": 8,
      "created_at": 1740700000000000
    }
  ]
}
```

### `amem://graph/important`

Return nodes with the highest decay scores (top 20). Useful for surfacing the most relevant and actively accessed memories.

**Format:** JSON object with count and node array (includes `decay_score` field).

```json
{
  "count": 20,
  "nodes": [
    {
      "id": 42,
      "event_type": "decision",
      "content": "Use PostgreSQL for the main database",
      "confidence": 0.95,
      "decay_score": 0.98,
      "session_id": 3
    }
  ]
}
```

## Resource Templates

### `amem://node/{id}`

Return a single cognitive event node with all its edges.

**Format:** JSON object with full node details, outgoing edges, and incoming edges.

```json
{
  "id": 42,
  "event_type": "decision",
  "content": "Use PostgreSQL for the main database",
  "confidence": 0.95,
  "session_id": 3,
  "created_at": 1740600000000000,
  "access_count": 12,
  "last_accessed": 1740700000000000,
  "decay_score": 0.98,
  "outgoing_edges": [
    {
      "target_id": 15,
      "edge_type": "caused_by",
      "weight": 1.0
    }
  ],
  "incoming_edges": [
    {
      "source_id": 55,
      "edge_type": "supports",
      "weight": 0.9
    }
  ]
}
```

### `amem://session/{id}`

Return all nodes belonging to a specific session.

**Format:** JSON object with session ID, node count, and node array.

```json
{
  "session_id": 3,
  "node_count": 18,
  "nodes": [
    {
      "id": 30,
      "event_type": "fact",
      "content": "PostgreSQL supports JSONB columns",
      "confidence": 0.9,
      "created_at": 1740600000000000
    }
  ]
}
```

### `amem://types/{type}`

Return all nodes of a specific event type. Valid types: `fact`, `decision`, `inference`, `correction`, `skill`, `episode`.

**Format:** JSON object with event type, count, and node array.

```json
{
  "event_type": "decision",
  "count": 32,
  "nodes": [
    {
      "id": 42,
      "content": "Use PostgreSQL for the main database",
      "confidence": 0.95,
      "session_id": 3,
      "created_at": 1740600000000000
    }
  ]
}
```

## Cross-Sister Resources

When running alongside other Agentra sisters, AgenticMemory resources can be referenced:

- Vision captures can link to `amem://node/{id}` for contextual memory
- Codebase analysis can reference `amem://session/{id}` for project knowledge
- Identity receipts can link to `amem://types/decision` for decision audit trails
- Time deadlines can reference `amem://graph/recent` for temporal context

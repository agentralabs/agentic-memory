---
status: stable
---

# Command Surface

Install commands are documented in [Installation](installation.md).

## Binaries

- `amem` (CLI engine)
- `agentic-memory-mcp` (MCP server)

## `amem` top-level

```bash
amem create
amem info
amem add
amem link
amem get
amem traverse
amem search
amem impact
amem resolve
amem sessions
amem export
amem import
amem decay
amem stats
amem quality
amem runtime-sync
amem budget
amem text-search
amem hybrid-search
amem centrality
amem path
amem revise
amem gaps
amem analogy
amem consolidate
amem drift
amem completions
```

## `agentic-memory-mcp` commands

```bash
agentic-memory-mcp serve
agentic-memory-mcp validate
agentic-memory-mcp info
agentic-memory-mcp delete
agentic-memory-mcp export
agentic-memory-mcp compact
agentic-memory-mcp stats
```

## New reliability commands

```bash
amem quality my_agent.amem
amem quality my_agent.amem --low-confidence 0.4 --stale-decay 0.2 --limit 25

amem runtime-sync my_agent.amem --workspace . --max-depth 4
amem runtime-sync my_agent.amem --workspace . --write-episode
amem budget my_agent.amem --horizon-years 20 --max-bytes 2147483648
```

## MCP Tools

All tools exposed by the `agentic-memory-mcp` MCP server:

### Core Tools

| Tool | Purpose |
|------|---------|
| `memory_add` | Add a new cognitive event to the memory graph |
| `memory_query` | Find memories matching conditions (pattern query) |
| `memory_quality` | Evaluate memory reliability: confidence, staleness, orphan nodes |
| `memory_traverse` | Walk the graph from a starting node, following edge types |
| `memory_correct` | Record a correction to a previous belief |
| `memory_resolve` | Follow the supersedes chain to get latest version of a belief |
| `memory_context` | Get the full context (subgraph) around a node |
| `memory_similar` | Find semantically similar memories using vector similarity |
| `memory_causal` | Impact analysis — find everything that depends on a given node |
| `memory_temporal` | Compare knowledge across two time periods |
| `memory_stats` | Get statistics about the memory graph |

### Context Capture Tools

| Tool | Purpose |
|------|---------|
| `conversation_log` | Log conversation context and auto-capture interactions |

### Grounding Tools (v0.3)

| Tool | Purpose |
|------|---------|
| `memory_ground` | Verify a claim has memory backing |
| `memory_evidence` | Get evidence for a claim from stored memories |
| `memory_suggest` | Find similar memories when a claim doesn't match exactly |

### Workspace Tools (v0.3)

| Tool | Purpose |
|------|---------|
| `memory_workspace_create` | Create a multi-memory workspace |
| `memory_workspace_add` | Add a .amem file to a workspace |
| `memory_workspace_list` | List loaded memories in a workspace |
| `memory_workspace_query` | Query across all loaded memories |
| `memory_workspace_compare` | Compare topics across memory contexts |
| `memory_workspace_xref` | Find topic distribution across contexts |

### Session Tools

| Tool | Purpose |
|------|---------|
| `session_start` | Start a new interaction session |
| `session_end` | End a session and optionally create episode summary |
| `memory_session_resume` | Load context from previous sessions (last summary, recent decisions, key facts) |

## V3 MCP Tools (v0.4)

V3 adds 13 immutable-capture MCP tools and 6 session resources.

### Capture Tools

| Tool | Purpose |
|------|---------|
| `memory_capture_message` | Capture user/assistant messages |
| `memory_capture_tool` | Capture tool calls with input/output |
| `memory_capture_file` | Capture file operations |
| `memory_capture_decision` | Capture decisions with reasoning |
| `memory_capture_boundary` | Capture session boundaries (compaction, etc.) |

### Retrieval Tools

| Tool | Purpose |
|------|---------|
| `memory_retrieve` | Smart context assembly with token budgeting |
| `memory_resurrect` | Restore full state at any timestamp |
| `memory_v3_session_resume` | Load context for session continuation |

### Search Tools

| Tool | Purpose |
|------|---------|
| `memory_search_temporal` | Search by time range |
| `memory_search_semantic` | Search by meaning/text |
| `memory_search_entity` | Search by file/person/entity |

### Stats Tools

| Tool | Purpose |
|------|---------|
| `memory_v3_stats` | Storage and index statistics |
| `memory_verify_integrity` | Cryptographic integrity verification |

### V3 MCP Resources

| Resource URI | Content |
|------|---------|
| `memory://v3/session/context` | Full session context |
| `memory://v3/session/decisions` | Recent decisions |
| `memory://v3/session/files` | Files modified |
| `memory://v3/session/errors` | Errors resolved |
| `memory://v3/session/activity` | Recent activity |
| `memory://v3/stats` | Storage statistics |

## MCP runtime capture controls

```bash
export AMEM_AUTO_CAPTURE_MODE=safe
export AMEM_AUTO_CAPTURE_REDACT=true
export AMEM_AUTO_CAPTURE_MAX_CHARS=2048
```

## Universal MCP entry

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "$HOME/.local/bin/agentic-memory-mcp",
      "args": ["serve"]
    }
  }
}
```

## Artifact

- Primary artifact: `.amem`

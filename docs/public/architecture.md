---
status: stable
---

# Architecture

AgenticMemory is a 4-crate Rust workspace with additional language bindings.

## Workspace Structure

```
agentic-memory/
  Cargo.toml                      (workspace root)
  crates/
    agentic-memory/               (core library)
    agentic-memory-mcp/           (MCP server)
    agentic-memory-cli/           (CLI binary: amem)
    agentic-memory-ffi/           (C FFI shared library)
  python/                         (Python SDK via PyO3)
  npm/wasm/                       (npm WASM package)
```

## Crate Responsibilities

### agentic-memory

The core library. All memory graph logic lives here.

- Node types: Facts, Decisions, Inferences, Corrections, Skills, Episodes
- Edge types: CausedBy, DerivedFrom, Supports, Contradicts, Supersedes, RelatedTo, PartOf, TemporalNext
- File format: `.amem` binary format (magic `AMEM`, version 1)
- Graph operations: add, query, traverse, correct, resolve, causal analysis
- Similarity search: cosine similarity with optional vector embeddings
- Session management: temporal chaining, auto-capture
- Grounding: claim verification against stored knowledge
- V3 Engine: append-only immortal storage with BLAKE3 integrity chains
- No I/O dependencies beyond file system access
- No MCP, CLI, or FFI concerns

### agentic-memory-mcp

The MCP server binary (`agentic-memory-mcp`).

- JSON-RPC 2.0 over stdio (default) and SSE/HTTP transport
- 25 core MCP tools (memory operations, grounding, workspaces, sessions)
- MCP resources via `amem://` URI scheme
- 4 MCP prompts (remember, reflect, correct, summarize)
- Auto-session lifecycle management (start on `initialized`, end on shutdown/EOF)
- Content-Length framing with 8 MiB frame limit
- Input validation: no silent fallback for invalid parameters
- Memory modes: minimal, smart, full

### agentic-memory-cli

The command-line interface binary (`amem`).

- Human-friendly terminal output
- All core operations exposed as subcommands
- Table, JSON, and text output formats
- Advanced queries: text-search, hybrid-search, centrality, path
- Reasoning tools: revise, gaps, analogy, drift
- Import/export workflows

### agentic-memory-ffi

C-compatible shared library for cross-language integration.

- Version function for compatibility checking
- Planned: opaque handle pattern for graph instances
- Planned: JSON-based data exchange for complex types

## Data Flow

```
Agent (Claude/GPT/etc.)
  |
  | MCP protocol (JSON-RPC 2.0 over stdio)
  v
agentic-memory-mcp
  |
  | Rust function calls
  v
agentic-memory (core)
  |
  | Binary I/O
  v
project.amem (file)
```

## File Format

The `.amem` binary format:

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic bytes: `AMEM` |
| 4 | 2 | Version: `0x0001` |
| 6 | 2 | Flags (reserved) |
| 8 | 8 | Node count |
| 16 | 8 | Edge count |
| 24 | 4 | Embedding dimension |
| 28 | ... | Node data (MessagePack encoded) |
| ... | ... | Edge index |

## Modules

| Module | Purpose |
|--------|---------|
| `graph` | In-memory graph structure, node/edge storage |
| `engine` | Query engine, write engine, similarity, traversal |
| `index` | Term index, temporal index, session index, type index |
| `types` | Core types: CognitiveEvent, Edge, EdgeType, EventType |
| `format` | `.amem` binary reader/writer, mmap support |
| `contracts` | Agentra contract validation |
| `v3` | V3 immortal architecture: append-only log, tiered storage |

## Cross-Sister Integration

AgenticMemory integrates with other Agentra sisters:

- **AgenticTime**: Decay curves inform memory freshness. Deadlines link to decision nodes via `atime://` URIs.
- **AgenticVision**: Visual captures link to memory nodes. Observation contexts stored as episodes.
- **AgenticCodebase**: Code analysis results stored as facts. Impact analysis feeds decision tracking.
- **AgenticIdentity**: Memory operations signed with identity receipts for audit trails.

## Runtime Isolation

Each project gets its own `.amem` file, resolved by deterministic path hashing. Same-name folders in different locations never share memory state. File-level locking with stale lock recovery ensures safe concurrent access.

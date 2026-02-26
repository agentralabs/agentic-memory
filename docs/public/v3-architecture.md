---
status: stable
---

# V3: Immortal Architecture

> **New in v0.4.0** — Memory that never dies.

V3 adds a complete append-only, content-addressed storage layer with BLAKE3 integrity chains.
Your agent's memory is now cryptographically tamper-proof, multi-client, and designed to last 20 years.

## Core Capabilities

- **Immortal Log** — Append-only storage with BLAKE3 integrity chains. Never deletes. Never modifies.
- **Five Indexes** — Temporal, semantic, causal, entity, and procedural. Find anything instantly.
- **Tiered Storage** — Hot, Warm, Cold, Frozen. 20 years of memory in approximately 500 MB.
- **Ghost Writer** — Auto-syncs to Claude, Cursor, Windsurf, and Cody. Zero configuration.
- **Smart Retrieval** — Multi-index fusion with token budgeting. Perfect context assembly.
- **Crash Recovery** — WAL with CRC32 checksums. Survives anything.

## Multi-Client Support

V3 is built for all AI agents, not just one.

| Client | Auto-Sync Location | Status |
|:---|:---|:---|
| **Claude Code** | `~/.claude/memory/V3_CONTEXT.md` | Full support |
| **Cursor** | `~/.cursor/memory/agentic-memory.md` | Full support |
| **Windsurf** | `~/.windsurf/memory/agentic-memory.md` | Full support |
| **Cody** | `~/.sourcegraph/cody/memory/agentic-memory.md` | Full support |

The Ghost Writer detects which clients are installed and syncs memory context to all of them
simultaneously. No configuration required — if the parent directory exists, the memory file is
created and kept up to date.

## Architecture Overview

```
+-------------------------------------------------------------+
|                     YOUR AI AGENT                           |
|           (Claude, Cursor, Windsurf, Cody)                  |
+----------------------------+--------------------------------+
                             |
                  +----------v----------+
                  |      MCP LAYER      |
                  |   Tools + Resources |
                  +----------+----------+
                             |
+----------------------------v--------------------------------+
|                        V3 ENGINE                            |
+--------------+--------------+--------------+----------------+
| Immortal Log | 5 Indexes    |Tiered Storage| Ghost Writer   |
| (append-only)| (T/S/C/E/P)  | (H/W/C/F)   | (multi-client) |
+--------------+--------------+--------------+----------------+
                             |
                  +----------v----------+
                  |     .amem FILE      |
                  |    (your memory)    |
                  +---------------------+
```

## The Five Indexes

Each index serves a different query pattern. Together they cover any way an agent
might need to recall information.

| Index | Purpose | Example Query |
|:---|:---|:---|
| **Temporal** | Find by time | "What happened yesterday?" |
| **Semantic** | Find by meaning | "Everything about contracts" |
| **Causal** | Find decision chains | "Why did we choose Rust?" |
| **Entity** | Find by file or person | "All changes to main.rs" |
| **Procedural** | Find workflows | "Steps to deploy" |

## Tiered Storage

Memory ages through four tiers automatically. Recent context stays fast;
historical context stays accessible at reduced cost.

| Tier | Age | Access Time | Storage |
|:---|:---|:---|:---|
| **Hot** | Less than 24 hours | Less than 1 ms | Memory |
| **Warm** | Less than 30 days | Less than 10 ms | Disk |
| **Cold** | Less than 1 year | Less than 100 ms | Compressed |
| **Frozen** | Forever | Less than 1 s | Archive |

## V3 MCP Tools

AgenticMemory V3 exposes **13 MCP tools** for AI agents.

### Capture Tools (5)

| Tool | Description |
|:---|:---|
| `memory_capture_message` | Capture user/assistant messages |
| `memory_capture_tool` | Capture tool calls with input/output |
| `memory_capture_file` | Capture file operations |
| `memory_capture_decision` | Capture decisions with reasoning |
| `memory_capture_boundary` | Capture session boundaries (compaction, etc.) |

### Retrieval Tools (3)

| Tool | Description |
|:---|:---|
| `memory_retrieve` | Smart context assembly with token budgeting |
| `memory_resurrect` | Restore full state at any timestamp |
| `memory_v3_session_resume` | Load context for session continuation |

### Search Tools (3)

| Tool | Description |
|:---|:---|
| `memory_search_temporal` | Search by time range |
| `memory_search_semantic` | Search by meaning/text |
| `memory_search_entity` | Search by file/person/entity |

### Stats Tools (2)

| Tool | Description |
|:---|:---|
| `memory_v3_stats` | Storage and index statistics |
| `memory_verify_integrity` | Cryptographic integrity verification |

### MCP Resources (6)

| Resource URI | Content |
|:---|:---|
| `memory://v3/session/context` | Full session context |
| `memory://v3/session/decisions` | Recent decisions |
| `memory://v3/session/files` | Files modified |
| `memory://v3/session/errors` | Errors resolved |
| `memory://v3/session/activity` | Recent activity |
| `memory://v3/stats` | Storage statistics |

## Auto-Capture Middleware

V3 includes an auto-capture middleware that intercepts MCP tool calls and
automatically records context without requiring explicit save calls.

- **Message capture**: User and assistant messages stored with role and token counts
- **Tool capture**: Tool calls with inputs, outputs, and duration
- **File operation detection**: Automatically detects file operations from tool names
- **Boundary events**: Session start, compaction, and context window management

## Migration from V2

V3 is fully backward compatible. Your V2 `.amem` files continue to work.

```bash
# V3 is enabled via feature flag in 0.4.0+
cargo build --features v3

# No migration command needed — it just works
```

| Aspect | V2 | V3 |
|:---|:---|:---|
| Session context | Manual `memory_session_resume` calls | Auto-injected via Ghost Writer |
| Capture scope | Decisions and facts only | Everything captured |
| Client support | Single-client (Claude) | Multi-client (Claude, Cursor, Windsurf, Cody) |
| Recall | Summary-based | Full procedural chains |
| Integrity | Basic | BLAKE3 cryptographic chains |

## Philosophy

AgenticMemory V3 is built on three principles:

1. **Capture everything** — Do not filter at write time. Every token matters.
2. **Index smart** — Five specialized indexes for any query pattern.
3. **Zero friction** — One install command. Works forever. No configuration.

The result: AI agents that never forget, never lose context, and can recall any moment
from any session.

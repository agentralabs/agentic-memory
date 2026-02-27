---
status: stable
---

# Configuration

AgenticMemory configuration options for all runtime modes.

## Environment Variables

| Variable | Default | Allowed Values | Effect |
|----------|---------|----------------|--------|
| `AMEM_BRAIN` | None | Path to `.amem` file | Explicit memory file path (overrides auto-detection) |
| `AGENTIC_TOKEN` | None | String | Bearer token for SSE server authentication |
| `RUST_LOG` | `info` | `trace`, `debug`, `info`, `warn`, `error` | Logging verbosity (standard tracing filter) |

## MCP Server Configuration

The MCP server (`agentic-memory-mcp`) accepts the following arguments:

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "~/.local/bin/agentic-memory-mcp-agentra",
      "args": ["serve"]
    }
  }
}
```

### Server Arguments

| Argument | Description |
|----------|-------------|
| `--memory <path>` / `-m <path>` | Path to `.amem` memory file |
| `--config <path>` / `-c <path>` | Configuration file path |
| `--log-level <level>` | Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`) |

### Server Subcommands

| Subcommand | Description |
|------------|-------------|
| `serve` | Start MCP server over stdio (default if no subcommand given) |
| `serve-http` | Start MCP server over HTTP/SSE (requires `sse` feature) |
| `validate` | Validate a memory file and print node/edge counts |
| `info` | Print server capabilities as JSON |
| `delete` | Delete a specific memory node by ID |
| `export` | Export all memories to stdout (json or csv) |
| `compact` | Remove low-scoring nodes (compaction) |
| `stats` | Print graph statistics |

## Memory Modes

The `--mode` flag controls how aggressively the server captures memories.

| Mode | Trigger | Behavior |
|------|---------|----------|
| `minimal` | `--mode minimal` | Save only when explicitly requested |
| `smart` | `--mode smart` (default) | Auto-save facts and decisions, skip transient chat |
| `full` | `--mode full` | Save everything potentially relevant |

## SSE Server Configuration

When using `serve-http` (requires the `sse` feature):

| Argument | Default | Description |
|----------|---------|-------------|
| `--addr` | `127.0.0.1:3000` | Listen address (host:port) |
| `--token` | None | Bearer token for authentication (also reads `AGENTIC_TOKEN`) |
| `--multi-tenant` | false | Enable per-user brain files |
| `--data-dir` | None | Directory for multi-tenant brain files (required with `--multi-tenant`) |

## Configuration File

Load a TOML configuration file with `--config`:

```toml
memory_path = "/path/to/brain.amem"
transport = "stdio"
sse_addr = "127.0.0.1:3000"
auto_save_interval = 30
log_level = "info"
```

| Field | Default | Description |
|-------|---------|-------------|
| `memory_path` | Auto-detected | Path to the `.amem` file |
| `transport` | `stdio` | Transport type: `stdio` or `sse` |
| `sse_addr` | `127.0.0.1:3000` | SSE listen address |
| `auto_save_interval` | `30` | Auto-save interval in seconds |
| `log_level` | `info` | Log level |

## File Location Resolution

AgenticMemory resolves the `.amem` file in this order:

1. Explicit `--memory` argument (if set)
2. `AMEM_BRAIN` environment variable (if set)
3. `.amem/brain.amem` in current directory
4. `~/.brain.amem` (global default)

## Compact Command

Remove nodes with low decay scores to reclaim space:

```bash
agentic-memory-mcp compact --memory brain.amem --keep-above 0.1 -y
```

| Argument | Description |
|----------|-------------|
| `--keep-above <threshold>` | Keep nodes with decay_score above this value |
| `-y, --yes` | Skip confirmation prompt |

## Export Command

Export all memories to stdout:

```bash
# JSON format (default)
agentic-memory-mcp export --memory brain.amem --format json

# CSV format
agentic-memory-mcp export --memory brain.amem --format csv
```

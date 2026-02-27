---
status: stable
---

# Troubleshooting

Common issues and solutions for AgenticMemory.

## Installation Issues

### Binary not found after install

Ensure `~/.local/bin` is in your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

### Install script fails with "jq not found"

The installer needs `jq` or `python3` for MCP config merging:

```bash
# macOS
brew install jq

# Ubuntu/Debian
sudo apt install jq

# Or use python3 (usually pre-installed)
python3 --version
```

### Cargo build fails

Ensure you have the latest stable Rust toolchain:

```bash
rustup update stable
```

If building with the `sse` feature fails, check that you have OpenSSL development headers:

```bash
# macOS (usually included)
brew install openssl

# Ubuntu/Debian
sudo apt install libssl-dev pkg-config
```

## MCP Server Issues

### Server not appearing in MCP client

1. Verify the binary exists: `ls ~/.local/bin/agentic-memory-mcp-agentra`
2. Check config was merged: look for `agentic-memory` in your MCP client config
3. Restart your MCP client completely (not just reload)
4. Run manually to check for errors: `agentic-memory-mcp serve`

### "AGENTIC_TOKEN required" error

This occurs in SSE server mode. Set the token:

```bash
export AGENTIC_TOKEN="$(openssl rand -hex 32)"
```

### Server crashes on startup

Check for stale lock files:

```bash
ls -la ~/.brain.amem.lock
# Remove stale locks (check PID first)
rm ~/.brain.amem.lock
```

### "Failed to read config file" error

Verify the config path exists and is valid TOML:

```bash
cat /path/to/config.toml
# Should be valid TOML format
```

## File Format Issues

### "Invalid memory file" error on validate

The file may be corrupted or not a valid `.amem` file:

```bash
agentic-memory-mcp validate --memory brain.amem
```

If validation fails, check file permissions and that the file was not truncated.

### File appears empty after operations

Ensure `save()` is called. The MCP server auto-saves at a configurable interval (default: 30 seconds). For CLI operations, changes are saved automatically after each command.

### Memory file grows too large

Use the compact command to remove low-scoring nodes:

```bash
# Preview what will be removed
agentic-memory-mcp compact --memory brain.amem --keep-above 0.1

# Apply compaction
agentic-memory-mcp compact --memory brain.amem --keep-above 0.1 -y
```

Or use the CLI budget estimator to plan ahead:

```bash
amem budget brain.amem --max-bytes 2147483648 --horizon-years 20
```

## Memory Graph Issues

### Nodes not appearing in query results

Check the sort order and filter parameters:

```bash
# Query with no filters to see all nodes
amem query brain.amem --limit 50

# Check if nodes exist in a specific session
amem sessions brain.amem
```

### memory_ground returns "ungrounded" for known facts

The grounding tool uses BM25 text search. Common causes:

1. **Phrasing mismatch:** Try different wording that matches the stored content more closely
2. **Threshold too high:** Lower the threshold parameter (default: 0.3)
3. **Node decayed:** Check if the node's decay score is very low with `amem stats`

### Corrections not reflected in queries

Use `memory_resolve` to follow the supersedes chain to the latest version:

```bash
amem resolve brain.amem 42
```

The original node still exists but is marked as superseded. Query results include both unless you filter by confidence.

### Traversal returns empty results

Check the direction and edge types:

```bash
# Try both directions
amem traverse brain.amem 42 --direction both

# Check what edges exist from the node
amem get brain.amem 42
```

## Workspace Issues

### "workspace not found" error

Workspaces are stored in `~/.agentic/memory/workspaces.json` for the CLI. Verify the workspace exists:

```bash
amem workspace list my-workspace
```

### Workspace query returns no results

Ensure `.amem` files were added to the workspace and the paths are valid:

```bash
amem workspace list my-workspace
# Check that file paths exist and are readable
```

## Performance Issues

### Slow startup with large .amem files

For graphs with more than 10,000 nodes, consider:

1. Compacting old nodes: `agentic-memory-mcp compact --memory brain.amem --keep-above 0.2 -y`
2. Splitting by project (use workspaces to query across them)
3. Running consolidation to merge duplicates: `amem consolidate brain.amem --deduplicate --confirm`

### High memory usage

The entire graph is loaded into memory. For very large memory files, split by project and use workspace queries for cross-project searches.

### Slow text search

BM25 search builds a term index on first query. Subsequent queries are fast. If the graph is very large, consider using `--type` filters to narrow the search scope.

## Session Issues

### Sessions not auto-starting

Auto-session starts when the MCP server receives the `initialized` notification. Ensure your MCP client sends this notification. If using `--mode minimal`, auto-session is skipped.

### Lost session data

The MCP server auto-saves periodically. If the server crashes before auto-save, some recent data may be lost. Reduce the auto-save interval in the config:

```toml
auto_save_interval = 10
```

### memory_session_resume returns empty

This means no previous session data exists (first session). Build up context by:

1. Using `memory_add` to store facts and decisions
2. Using `session_end` with a summary to create episode nodes
3. Subsequent calls to `memory_session_resume` will return this context

## Getting Help

- GitHub Issues: https://github.com/agentralabs/agentic-memory/issues
- Documentation: https://agentralabs.tech/docs/memory

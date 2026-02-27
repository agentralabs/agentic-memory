---
status: stable
---

# CLI Reference

The `amem` CLI provides command-line access to AgenticMemory graph-based memory files.

## Global Options

| Option | Description |
|--------|-------------|
| `--format <fmt>` | Output format: `text` (default), `json` |
| `--verbose` | Enable debug logging |
| `-h, --help` | Print help information |
| `-V, --version` | Print version |

Running `amem` with no subcommand launches an interactive REPL.

## Commands

### `amem init`

Create a new empty `.amem` file.

```bash
# Create with default 128-dimension vectors
amem init project.amem

# Create with custom vector dimension
amem init project.amem --dimension 256
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | path | Yes | Path to the `.amem` file to create |
| `--dimension` | integer | No | Feature vector dimension (default: 128) |

Alias: `amem create`

### `amem info`

Display information about an `.amem` file.

```bash
amem info project.amem
```

### `amem add`

Add a cognitive event to the graph.

```bash
# Add a fact
amem add project.amem fact "Rust uses zero-cost abstractions"

# Add a decision with confidence
amem add project.amem decision "Use PostgreSQL for persistence" --confidence 0.95

# Add a correction referencing a previous node
amem add project.amem correction "Actually uses SQLite" --supersedes 42
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | path | Yes | Path to the `.amem` file |
| `type` | string | Yes | Event type: `fact`, `decision`, `inference`, `correction`, `skill`, `episode` |
| `content` | string | Yes | The content text |
| `--session` | integer | No | Session ID (default: 0) |
| `--confidence` | float | No | Confidence 0.0-1.0 (default: 1.0) |
| `--supersedes` | integer | No | Node ID being corrected (for corrections) |

### `amem link`

Add an edge between two nodes.

```bash
amem link project.amem 1 2 supports --weight 0.9
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | path | Yes | Path to the `.amem` file |
| `source_id` | integer | Yes | Source node ID |
| `target_id` | integer | Yes | Target node ID |
| `edge_type` | string | Yes | Edge type: `caused_by`, `derived_from`, `supports`, `contradicts`, `supersedes`, `related_to`, `part_of`, `temporal_next` |
| `--weight` | float | No | Edge weight 0.0-1.0 (default: 1.0) |

### `amem get`

Get a specific node by ID.

```bash
amem get project.amem 42
```

### `amem traverse`

Run a traversal query from a starting node.

```bash
# Traverse backward from node 42
amem traverse project.amem 42 --direction backward

# Follow only specific edge types
amem traverse project.amem 42 --edge-types supports,caused_by --max-depth 3
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | path | Yes | Path to the `.amem` file |
| `start_id` | integer | Yes | Starting node ID |
| `--edge-types` | string | No | Comma-separated edge types to follow |
| `--direction` | string | No | `forward`, `backward`, or `both` (default: `backward`) |
| `--max-depth` | integer | No | Maximum traversal depth (default: 5) |
| `--max-results` | integer | No | Maximum nodes to return (default: 50) |
| `--min-confidence` | float | No | Minimum confidence filter (default: 0.0) |

### `amem query`

Pattern query -- find nodes matching conditions.

```bash
# Find recent decisions
amem query project.amem --type decision --sort recent

# Find high-confidence facts from a specific session
amem query project.amem --type fact --session 3 --min-confidence 0.9
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | path | Yes | Path to the `.amem` file |
| `--type` | string | No | Comma-separated event types to filter |
| `--session` | string | No | Comma-separated session IDs |
| `--min-confidence` | float | No | Minimum confidence |
| `--max-confidence` | float | No | Maximum confidence |
| `--after` | integer | No | Created after (Unix microseconds) |
| `--before` | integer | No | Created before (Unix microseconds) |
| `--sort` | string | No | `recent`, `confidence`, `accessed`, `importance` (default: `recent`) |
| `--limit` | integer | No | Maximum results (default: 20) |

Alias: `amem search`

### `amem impact`

Run causal impact analysis on a node.

```bash
amem impact project.amem 42 --max-depth 5
```

### `amem resolve`

Follow the SUPERSEDES chain to find the latest version of a node.

```bash
amem resolve project.amem 42
```

### `amem sessions`

List all sessions in the file.

```bash
amem sessions project.amem --limit 10
```

### `amem export`

Export the graph as JSON.

```bash
# Export everything
amem export project.amem --pretty

# Export only nodes from session 3
amem export project.amem --session 3 --nodes-only
```

### `amem import`

Import nodes and edges from a JSON file.

```bash
amem import project.amem data.json
```

### `amem ground`

Verify a claim has memory backing.

```bash
amem ground project.amem "Rust uses zero-cost abstractions" --threshold 0.3
```

### `amem evidence`

Return supporting evidence for a query.

```bash
amem evidence project.amem "database migration" --limit 5
```

### `amem suggest`

Suggest similar memories for a phrase.

```bash
amem suggest project.amem "authentication module" --limit 5
```

### `amem text-search`

BM25 text search over node contents.

```bash
amem text-search project.amem "deploy pipeline" --type fact,decision --limit 10
```

### `amem hybrid-search`

Combined BM25 + vector search with RRF fusion.

```bash
amem hybrid-search project.amem "authentication flow" --text-weight 0.6 --vec-weight 0.4
```

### `amem centrality`

Compute node importance scores.

```bash
# PageRank (default)
amem centrality project.amem --limit 10

# Degree centrality
amem centrality project.amem --algorithm degree

# Betweenness centrality
amem centrality project.amem --algorithm betweenness
```

### `amem path`

Find shortest path between two nodes.

```bash
amem path project.amem 1 42 --direction both --max-depth 10
```

### `amem revise`

Belief revision -- counterfactual analysis.

```bash
amem revise project.amem "The API uses REST" --threshold 0.6 --confidence 0.9
```

### `amem gaps`

Reasoning gap detection.

```bash
amem gaps project.amem --threshold 0.5 --sort dangerous
```

### `amem analogy`

Find structurally similar past situations.

```bash
amem analogy project.amem "migrating from MySQL to PostgreSQL" --limit 5
```

### `amem consolidate`

Brain maintenance -- consolidation.

```bash
# Dry-run all operations
amem consolidate project.amem --all

# Apply deduplication with custom threshold
amem consolidate project.amem --deduplicate --threshold 0.90 --confirm

# Link contradictions and promote stable inferences
amem consolidate project.amem --link-contradictions --promote-inferences --confirm
```

### `amem drift`

Track how beliefs about a topic evolved over time.

```bash
amem drift project.amem "deployment strategy" --limit 5
```

### `amem decay`

Run decay calculations and report stale nodes.

```bash
amem decay project.amem --threshold 0.1
```

### `amem stats`

Print detailed graph statistics.

```bash
amem stats project.amem
# Output:
#   Brain: project.amem
#     Nodes:    142
#     Edges:    215
#     Sessions: 8
#     File:     12.4 KB
#     Types:
#       facts: 85
#       decisions: 32
#       inferences: 15
#       episodes: 10
```

### `amem quality`

Graph health and memory quality report.

```bash
amem quality project.amem --low-confidence 0.45 --stale-decay 0.20
```

### `amem runtime-sync`

Scan workspace artifacts and optionally write an episode snapshot.

```bash
amem runtime-sync project.amem --workspace /path/to/project --write-episode
```

### `amem budget`

Estimate long-horizon storage usage against a fixed budget.

```bash
amem budget project.amem --max-bytes 2147483648 --horizon-years 20
```

### `amem workspace`

Workspace operations across multiple memory files.

```bash
# Create a workspace
amem workspace create my-workspace

# Add memory files
amem workspace add my-workspace project-a.amem --role primary --label "Project A"
amem workspace add my-workspace project-b.amem --role secondary --label "Project B"

# List files in a workspace
amem workspace list my-workspace

# Query across all files
amem workspace query my-workspace "authentication" --limit 10

# Compare a topic across contexts
amem workspace compare my-workspace "database schema" --limit 5

# Cross-reference
amem workspace xref my-workspace "deployment"
```

### `amem completions`

Generate shell completion scripts.

```bash
amem completions bash > ~/.local/share/bash-completion/completions/amem
amem completions zsh > ~/.zfunc/_amem
amem completions fish > ~/.config/fish/completions/amem.fish
```

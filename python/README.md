# AgenticMemory Python SDK (v0.2.0)

Python SDK for AgenticMemory -- portable binary graph memory for AI agents. 16 query types, zero dependencies.

## Install

```bash
pip install agentic-brain
```

### With LLM integrations

```bash
pip install agentic-brain[anthropic]   # Claude
pip install agentic-brain[openai]      # GPT
pip install agentic-brain[ollama]      # Local models
pip install agentic-brain[all]         # All providers
```

## Quick Start

```python
from agentic_memory import Brain

brain = Brain("my_agent.amem")
brain.add_fact("User is a Python developer", session=1)
brain.add_decision("Recommended FastAPI for REST APIs", session=1)

print(brain.facts())
print(brain.info())
```

## v0.2 Query Expansion

Nine new methods added in v0.2.0:

```python
brain = Brain("my_agent.amem")

# Retrieval
results = brain.search_text("API rate limit")           # BM25 (1.58 ms @ 100K)
results = brain.search("caching strategy", top_k=10)    # Hybrid BM25+vector (10.83 ms)

# Structural analysis
scores = brain.centrality(metric="pagerank")             # PageRank (34.3 ms @ 100K)
path   = brain.shortest_path(src=42, dst=99)             # BFS (104 us @ 100K)

# Cognitive reasoning
report  = brain.revise(node_id=42)                       # Counterfactual cascade (53.4 ms)
gaps    = brain.gaps()                                    # Find reasoning weaknesses
matches = brain.analogy(node_id=42, top_k=5)             # Structural pattern matching

# Graph maintenance
report = brain.consolidate(dry_run=True)                 # Dedup, contradiction linking
drift  = brain.drift()                                   # Belief evolution tracking (68.4 ms)
```

## With LLM Integration

```python
from agentic_memory import Brain, MemoryAgent
from agentic_memory.integrations import AnthropicProvider

brain = Brain("my_agent.amem")
agent = MemoryAgent(brain, AnthropicProvider())

response = agent.chat("My name is Alice. I work on ML systems.", session=1)
response = agent.chat("What do I work on?", session=2)
```

## Test Coverage

104 tests across 8 modules, including 20 tests for the v0.2 query expansion methods.

## Requirements

- Python >= 3.10
- `amem` binary (Rust core engine) -- install via `cargo install amem`

## Documentation

- [API Reference](../docs/api-reference.md)
- [Integration Guide](../docs/integration-guide.md)
- [Benchmarks](../docs/benchmarks.md)
- [Full README](../README.md)

## License

MIT

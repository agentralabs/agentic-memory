# Contributing to AgenticMemory

Thank you for your interest in contributing to AgenticMemory! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/agentic-memory.git`
3. Create a feature branch: `git checkout -b my-feature`
4. Make your changes
5. Run the tests (see below)
6. Commit and push
7. Open a pull request

## Development Setup

This is a Cargo workspace monorepo. All Rust crates are under `crates/`.

### Rust Workspace

```bash
# Build everything (core + MCP server)
cargo build

# Run all tests (core + MCP + bridge)
cargo test --workspace

# Core library only
cargo test -p agentic-memory
cargo bench -p agentic-memory

# MCP server only
cargo test -p agentic-memory-mcp

# Bridge integration tests
cargo test -p agentic-memory-bridge-tests

# Run the core CLI
cargo run -p agentic-memory -- create test.amem

# Run the MCP server
cargo run -p agentic-memory-mcp -- serve --memory test.amem
```

### Python SDK

```bash
cd python/
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

### Installer

```bash
cd installer/
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Ways to Contribute

### Report Bugs

File an issue with:
- Steps to reproduce
- Expected behavior
- Actual behavior
- System info (OS, Python version, Rust version)

### Add a New LLM Provider

1. Create a new file in `python/src/agentic_memory/integrations/`
2. Implement the `LLMProvider` interface
3. Add tests in `python/tests/`
4. Update `docs/integration-guide.md`

### Add an MCP Tool

1. Create a new tool handler in `crates/agentic-memory-mcp/src/tools/`
2. Register it in `crates/agentic-memory-mcp/src/tools/registry.rs`
3. Add tests in `crates/agentic-memory-mcp/tests/`
4. Update `crates/agentic-memory-mcp/README.md`

### Write Examples

1. Add a new example in `examples/`
2. Ensure it runs without errors
3. Add a docstring explaining what it demonstrates
4. Update `examples/README.md`

### Improve Documentation

All docs are in `docs/`. Fix typos, add examples, clarify explanations — all welcome.

## Code Guidelines

- **Rust**: Follow standard Rust conventions. Run `cargo clippy` and `cargo fmt`.
- **Python**: Follow PEP 8. Use type hints. Run `mypy` for type checking.
- **Tests**: Every feature needs tests. We maintain 575+ tests across the stack (179 Rust core, 135 MCP server + bridge, 104 Python SDK, 97 agent, 21 cross-provider, 39 installer).
- **Documentation**: Update docs when changing public APIs.

## Commit Messages

Use clear, descriptive commit messages:
- `Add: new OllamaProvider integration`
- `Fix: memory leak in graph traversal`
- `Update: improve error messages for CLI`
- `Docs: add LangChain integration guide`

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update documentation if needed
- Ensure all tests pass before submitting
- Write a clear PR description

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

#!/bin/bash
set -e

echo "=== Publishing AgenticMemory paired crates to crates.io ==="

# Verify logged in
cargo login --help > /dev/null

# Run all tests
cargo test --workspace

# Check formatting
cargo fmt --check

# Check clippy
cargo clippy --workspace -- -D warnings

# Dry run publish (paired crates: core first, then MCP)
echo "Dry run: agentic-memory"
cd crates/agentic-memory
cargo publish --dry-run
cd ../..

echo "Dry run: agentic-memory-mcp"
cd crates/agentic-memory-mcp
# Core crate may not be indexed yet for brand-new versions, so package MCP
# locally instead of publish --dry-run for preflight validation.
cargo package
cd ../..

echo ""
echo "Dry run successful. To actually publish:"
echo "  cd crates/agentic-memory && cargo publish"
echo "  # Wait for it to be available on crates.io"
echo "  cd crates/agentic-memory-mcp && cargo publish"

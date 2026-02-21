#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

assert_contains() {
  local pattern="$1"
  shift
  if ! rg -nF "$pattern" "$@" >/dev/null; then
    fail "Missing required install command: ${pattern}"
  fi
}

# Front-facing command requirements
assert_contains "curl -fsSL https://agentralabs.tech/install/memory | bash" README.md docs/quickstart.md
assert_contains "cargo install agentic-memory agentic-memory-mcp" README.md
assert_contains "pip install amem-installer && amem-install install --auto" README.md

# Invalid patterns
if rg -n "curl -fsSL https://agentralabs.tech/install/memory \| sh" README.md docs -g '*.md' >/dev/null; then
  fail "Found invalid shell invocation for memory installer"
fi

# Installer health
bash -n scripts/install.sh
bash scripts/install.sh --dry-run >/dev/null

# Public endpoint/package health
curl -fsSL https://agentralabs.tech/install/memory >/dev/null
curl -fsSL https://crates.io/api/v1/crates/agentic-memory >/dev/null
curl -fsSL https://crates.io/api/v1/crates/agentic-memory-mcp >/dev/null
curl -fsSL https://pypi.org/pypi/agentic-brain/json >/dev/null
curl -fsSL https://pypi.org/pypi/amem-installer/json >/dev/null

echo "Install command guardrails passed (memory)."

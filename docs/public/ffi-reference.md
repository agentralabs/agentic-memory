---
status: stable
---

# FFI Reference

AgenticMemory exposes a minimal C-compatible FFI layer through the `agentic-memory-ffi` crate. This enables integration from any language that supports C function calls (Python ctypes, Node.js ffi-napi, Ruby FFI, Go cgo, etc.).

## Shared Library

Build the shared library:

```bash
cargo build --release -p agentic-memory-ffi
# Output: target/release/libagentic_memory_ffi.{so,dylib,dll}
```

## Functions

### `agentic_memory_ffi_version`

Return the crate version string.

```c
const char* agentic_memory_ffi_version(void);
```

**Returns:** Static version string (e.g., `"0.1.0"`). Caller must NOT free.

## Current Status

The FFI crate currently exposes version information only. The full FFI surface is under active development. For programmatic access today, use either:

- The **MCP server** (`agentic-memory-mcp`) for agent integration via the Model Context Protocol
- The **Rust API** directly via the `agentic-memory` crate (see `docs/public/rust-api.md`)
- The **CLI** (`amem`) for shell scripting and automation

## Planned FFI Surface

The following functions are planned for the FFI layer:

| Function | Description |
|----------|-------------|
| `amem_open` | Open or create a memory graph file |
| `amem_create` | Create a new empty memory graph |
| `amem_close` | Close and free a memory graph handle |
| `amem_save` | Persist in-memory changes to disk |
| `amem_stats` | Get graph statistics as a JSON string |
| `amem_add_node` | Add a cognitive event node |
| `amem_add_edge` | Add an edge between nodes |
| `amem_get_node` | Get a node by ID as JSON |
| `amem_query` | Run a pattern query, return JSON |
| `amem_traverse` | Walk the graph from a starting node |
| `amem_ground` | Verify a claim has memory backing |
| `amem_free_string` | Free a string returned by other FFI functions |
| `amem_last_error` | Get the last error message |

## Example: Python ctypes (version check)

```python
import ctypes

lib = ctypes.CDLL("libagentic_memory_ffi.dylib")

lib.agentic_memory_ffi_version.restype = ctypes.c_char_p
lib.agentic_memory_ffi_version.argtypes = []

version = lib.agentic_memory_ffi_version()
print(f"AgenticMemory FFI version: {version.decode()}")
```

## Thread Safety

The FFI layer is designed to be thread-safe. Once the full surface is implemented, all functions will be safe to call with different handles from multiple threads. Concurrent access to the same handle will require external synchronization.

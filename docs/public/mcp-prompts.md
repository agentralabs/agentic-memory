---
status: stable
---

# MCP Prompts

AgenticMemory provides 4 built-in MCP prompts that agents can invoke for structured memory operations.

## `remember`

Guide for storing new information in memory.

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `information` | string | Yes | What to remember |
| `context` | string | No | Why this is important |

### Behavior

The prompt instructs the agent to:

1. Analyze the information to determine the appropriate event type (fact, decision, inference, skill)
2. Identify any existing memories this might relate to or contradict
3. Use the `memory_add` tool to store the information with appropriate edges

### Example

```json
{
  "name": "remember",
  "arguments": {
    "information": "The API rate limit is 100 requests per minute",
    "context": "Discovered during load testing"
  }
}
```

## `reflect`

Guide for understanding past decisions and reasoning.

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `topic` | string | Yes | What decision or belief to reflect on |
| `node_id` | integer | No | Specific node ID to start from |

### Behavior

The prompt instructs the agent to:

1. Use `memory_query` to find relevant decisions or beliefs
2. Use `memory_traverse` with `direction="backward"` to find the reasoning chain
3. Use `memory_causal` to understand dependencies
4. Summarize the reasoning chain clearly

### Example

```json
{
  "name": "reflect",
  "arguments": {
    "topic": "Why did we choose PostgreSQL over MongoDB?"
  }
}
```

## `correct`

Guide for updating beliefs and correcting past information.

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `old_belief` | string | Yes | What was previously believed |
| `new_information` | string | Yes | The correct information |
| `reason` | string | No | Why this is being corrected |

### Behavior

The prompt instructs the agent to:

1. Use `memory_query` to find the node containing the old belief
2. Use `memory_causal` to see what depends on this belief
3. Use `memory_correct` to create the correction
4. Consider if dependent decisions should also be corrected

### Example

```json
{
  "name": "correct",
  "arguments": {
    "old_belief": "The API uses basic auth",
    "new_information": "The API uses OAuth 2.0 with PKCE",
    "reason": "Updated during security audit"
  }
}
```

## `summarize`

Guide for creating a session summary.

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `session_id` | integer | No | Session ID to summarize (defaults to current) |

### Behavior

The prompt loads all nodes from the target session and instructs the agent to:

1. Review all memories in the session
2. Identify the main topic or goal
3. Extract key facts learned
4. Note important decisions made
5. Record any corrections
6. Summarize the outcome
7. Use `session_end` with `create_episode=true` and the summary

### Example

```json
{
  "name": "summarize",
  "arguments": {
    "session_id": 5
  }
}
```

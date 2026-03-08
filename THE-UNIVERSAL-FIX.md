# THE UNIVERSAL FIX
## Memory That Works With ANY System — Not Just Hydra

> **Date:** March 2026  
> **Principle:** The fix lives INSIDE agentic-memory-mcp. Not inside Hydra. Not inside Claude Code. Not inside any host. The memory server is self-sufficient.  
> **Standard:** `pip install agentic-brain && amem install --auto` → memory works. On any client. Forever.

---

## 1. THE ARCHITECTURAL INSIGHT

The Hydra fix is valid for Hydra. But here's the problem with making it Hydra-only:

```
IF THE FIX LIVES IN HYDRA:
══════════════════════════
  ✅ Hydra users get memory
  ❌ Claude Code users: nothing captured
  ❌ Cursor users: nothing captured
  ❌ Windsurf users: nothing captured
  ❌ OpenClaw users: nothing captured
  ❌ Codex users: nothing captured
  ❌ Cody users: nothing captured
  ❌ Ollama users: nothing captured
  ❌ Any future MCP client: nothing captured

  That's not a memory solution. That's a Hydra feature.
```

The fix must live where it's UNIVERSAL — inside `agentic-memory-mcp` itself. The MCP server is a **process running on the user's machine.** It doesn't need permission from the host to watch files, run daemons, write context, or manage storage. It's a sovereign process. It should ACT like one.

```
IF THE FIX LIVES IN AGENTIC-MEMORY-MCP:
════════════════════════════════════════
  ✅ Claude Code: captured (log watcher + Ghost Writer)
  ✅ Cursor: captured (log watcher + Ghost Writer)
  ✅ Windsurf: captured (log watcher + Ghost Writer)
  ✅ Cody: captured (log watcher + Ghost Writer)
  ✅ OpenClaw: captured (tool-call passthrough capture)
  ✅ Codex: captured (log watcher + Ghost Writer)
  ✅ Ollama: captured (proxy mode or tool calls)
  ✅ Hydra: captured (native integration + all the above)
  ✅ Any future MCP client: captured (Ghost Writer is universal)
  
  That's a memory solution. Install once. Works everywhere.
```

---

## 2. THE SELF-SUFFICIENT MCP SERVER

When `agentic-memory-mcp serve` starts, it doesn't just sit there waiting for tool calls. It becomes an **active cognitive process** that captures, consolidates, injects, and protects memory — regardless of what the host client does or doesn't do.

```
WHAT HAPPENS WHEN THE MCP SERVER STARTS:
════════════════════════════════════════

  agentic-memory-mcp serve
       │
       ├── 1. INITIALIZE STORAGE
       │     ├── Open or create .amem (hot path)
       │     ├── Open or create .longevity.db (cold path)
       │     └── Run schema migrations if needed
       │
       ├── 2. SPAWN CAPTURE DAEMON (tokio::spawn)
       │     ├── Detect which client started us
       │     │   (check parent process, env vars, stdio patterns)
       │     ├── Find conversation log directory for that client
       │     │   ├── Claude Code: ~/.claude/projects/*/conversations/
       │     │   ├── Cursor: .cursor/ in workspace
       │     │   ├── VS Code: extension storage
       │     │   ├── Windsurf: ~/.windsurf/ 
       │     │   ├── Cody: ~/.sourcegraph/cody/
       │     │   └── Generic: AMEM_WATCH_DIR env var
       │     ├── Start file watcher (notify::RecommendedWatcher)
       │     ├── Start dedup engine (BLAKE3 content-addressed)
       │     └── Log: "Capture active — watching {path}"
       │
       ├── 3. SPAWN CONSOLIDATION SCHEDULER (tokio::spawn)
       │     ├── Check time since last consolidation
       │     ├── If overdue → run immediately
       │     ├── Schedule nightly Raw → Episode
       │     ├── Schedule weekly Episode → Summary
       │     └── Schedule monthly Summary → Pattern
       │
       ├── 4. SPAWN BACKUP SCHEDULER (tokio::spawn) 
       │     ├── If backup configured → run on schedule
       │     └── If not configured → skip (user can configure later)
       │
       ├── 5. GENERATE GHOST WRITER CONTEXT
       │     ├── Compute session context from longevity store
       │     ├── Include CRITICAL INSTRUCTION block
       │     ├── Include user profile + recent decisions
       │     ├── Write to ALL detected client context paths:
       │     │   ├── ~/.claude/memory/V3_CONTEXT.md
       │     │   ├── ~/.cursor/memory/agentic-memory.md
       │     │   ├── ~/.windsurf/memory/agentic-memory.md
       │     │   ├── ~/.sourcegraph/cody/memory/agentic-memory.md
       │     │   └── Project-local .amem-context.md
       │     └── Start 5-second sync loop (refresh context periodically)
       │
       ├── 6. REGISTER MCP TOOLS (existing + new longevity tools)
       │     ├── 13 V3 capture/retrieval/search tools
       │     ├── 8 V4 longevity tools
       │     └── 1 new: memory_capabilities
       │
       └── 7. START MCP STDIO LOOP
             ├── Listen for JSON-RPC tool calls
             ├── On ANY tool call: capture the call context
             │   (even non-memory tools reveal what the user is doing)
             └── Handle tools as normal
```

**That's it.** The MCP server is now a fully autonomous memory process. It captures conversations by watching the client's log files. It injects context by writing Ghost Writer files. It compresses memory on schedule. It backs up on schedule. The host client doesn't need to do ANYTHING special. It doesn't even need to know AgenticMemory exists beyond having it listed in its MCP config.

---

## 3. THE FOUR UNIVERSAL CAPTURE CHANNELS

Every capture channel works independently. Any ONE is sufficient. Together they're bulletproof.

### Channel A: Tool-Call Passthrough Capture

```
WORKS WITH: Every MCP client (it's the MCP protocol itself)
HOW:        When ANY memory tool is called, capture the full context.
DEPENDS ON: LLM deciding to call memory tools (unreliable — the 11philip22 problem)
ROLE:       Best-effort. Captures rich structured data when it works.
            NOT the primary capture path.

ENHANCEMENT: On every tool call (not just memory tools), capture metadata:
  - Which tool was called (reveals what user is doing)
  - Timestamp (reveals session activity pattern)
  - Input size (reveals complexity of task)
  
  This is passive intelligence — the MCP server SEES every tool call
  that routes through it, even if the LLM isn't explicitly saving memories.
```

### Channel B: Client Log File Monitoring

```
WORKS WITH: Claude Code, Cursor, VS Code, Windsurf, Cody, any client that 
            writes conversation logs to disk
HOW:        notify::RecommendedWatcher on the client's conversation directory
DEPENDS ON: Client writing logs to a known path (most do)
ROLE:       PRIMARY capture path. Zero LLM dependency.

CLIENT LOG PATHS (detected automatically):
  Claude Code:  ~/.claude/projects/{hash}/conversations/*.jsonl
  Cursor:       {workspace}/.cursor/conversations/ or ~/.cursor/
  VS Code:      ~/.vscode/data/ (extension-specific)
  Windsurf:     ~/.windsurf/conversations/
  Cody:         ~/.sourcegraph/cody/conversations/
  Generic:      $AMEM_WATCH_DIR (user-configurable)

DETECTION LOGIC:
  On startup, check parent process name and environment:
  - CLAUDE_PROJECT_DIR set? → Claude Code
  - CURSOR_WORKSPACE set? → Cursor
  - VSCODE_PID set? → VS Code
  - Check common paths and use first that exists
  - Fall back to $AMEM_WATCH_DIR if set
  - If nothing detected → log warning, rely on Channels A and D
```

### Channel C: Ghost Writer Instruction (Context Injection)

```
WORKS WITH: Every client that reads context/memory files on startup
            (Claude Code, Cursor, Windsurf, Cody — all confirmed)
HOW:        Ghost Writer writes a context markdown file that INSTRUCTS
            the LLM to call memory_capture_message on every exchange
DEPENDS ON: LLM reading and following the Ghost Writer file
ROLE:       Force multiplier for Channel A. Turns unreliable LLM
            tool calls into systematic capture by telling the LLM
            it MUST call capture tools.

THE GHOST WRITER FILE CONTAINS:
  1. CRITICAL INSTRUCTION: "Call memory_capture_message on every user 
     message. This is mandatory. Takes <1ms."
  2. User profile: Who this person is, their expertise, preferences
  3. Recent context: What they were working on in the last session
  4. Active decisions: What's been decided but not yet acted on
  5. Memory capabilities: What the LLM can truthfully claim about memory
  6. Token budget: Entire file fits within 4K tokens

WHY THIS WORKS EVEN WITHOUT THE FILE WATCHER:
  If the LLM follows the CRITICAL INSTRUCTION and calls 
  memory_capture_message on every message, Channel A catches 
  everything. The Ghost Writer CREATES the behavior that makes
  Channel A reliable.
  
  Channel B (file watcher) is the safety net if the LLM ignores 
  the instruction. Together they're defense in depth.
```

### Channel D: Proxy Mode (Advanced — Future)

```
WORKS WITH: Any LLM client, any LLM API, even non-MCP systems
HOW:        amem-proxy sits between client and LLM API
            Intercepts all requests and responses bidirectionally
DEPENDS ON: User configuring proxy (not zero-config)
ROLE:       Nuclear option. Captures EVERYTHING regardless of client,
            protocol, or MCP support. For power users and enterprise.

NOT IN V4.0 — documented for V4.2+
```

### How The Channels Complement Each Other

```
SCENARIO: Claude Code user, normal conversation
  Channel A: LLM calls memory_capture (Ghost Writer told it to) ✅
  Channel B: File watcher captures from Claude conversation JSONL ✅
  Channel C: Ghost Writer injected context on startup ✅
  Dedup: Same messages from A and B → stored once
  RESULT: 100% capture

SCENARIO: Cursor user, LLM ignores memory tools  
  Channel A: LLM doesn't call tools ❌
  Channel B: File watcher captures from Cursor logs ✅
  Channel C: Ghost Writer was read but LLM ignored instruction ❌  
  RESULT: 100% capture (Channel B alone is sufficient)

SCENARIO: OpenClaw user, no conversation log files
  Channel A: LLM calls memory_capture (Ghost Writer told it to) ✅
  Channel B: No log files to watch ❌
  Channel C: Ghost Writer context injected ✅
  RESULT: Capture depends on LLM following instruction
          → This is the weakest scenario. Document it.
          → Recommend user set $AMEM_WATCH_DIR or use proxy mode.

SCENARIO: Ollama local user, raw API, no MCP
  Channel A: No MCP ❌
  Channel B: No conversation files ❌
  Channel C: No Ghost Writer mechanism ❌
  Channel D: Proxy mode captures API traffic ✅
  RESULT: Proxy mode is the only option for non-MCP clients.
          → V4.2+ feature. Documented as requirement.

SCENARIO: Unknown future MCP client (2028)
  Channel A: MCP tools work ✅ (it's the protocol)
  Channel B: Maybe has log files, maybe not → best effort
  Channel C: Ghost Writer writes to project-local .amem-context.md ✅
             (client may or may not read it, but it's there)
  RESULT: At minimum Channel A works. Ghost Writer is there if 
          the client supports reading context files.
```

---

## 4. THE UNIVERSAL GHOST WRITER

The Ghost Writer is the ONE mechanism that works across ALL clients. It's a file. Every client reads files. The file tells the LLM what it knows and what it must do.

### 4.1 Multi-Client Write Strategy

```rust
/// Ghost Writer writes to EVERY detected client path simultaneously.
/// This is not wasteful — it's insurance. 

pub struct UniversalGhostWriter {
    /// All paths to write context to
    write_paths: Vec<PathBuf>,
    /// Longevity store for context computation
    store: Arc<LongevityStore>,
    /// Project identifier
    project_id: String,
}

impl UniversalGhostWriter {
    pub fn detect_all_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        let home = dirs::home_dir().unwrap();
        
        // Claude Code
        let claude_path = home.join(".claude/memory/V3_CONTEXT.md");
        if claude_path.parent().map(|p| p.exists()).unwrap_or(false) {
            paths.push(claude_path);
        }
        
        // Cursor
        let cursor_path = home.join(".cursor/memory/agentic-memory.md");
        if cursor_path.parent().map(|p| p.exists()).unwrap_or(false) {
            paths.push(cursor_path);
        }
        
        // Windsurf
        let windsurf_path = home.join(".windsurf/memory/agentic-memory.md");
        if windsurf_path.parent().map(|p| p.exists()).unwrap_or(false) {
            paths.push(windsurf_path);
        }
        
        // Cody
        let cody_path = home.join(".sourcegraph/cody/memory/agentic-memory.md");
        if cody_path.parent().map(|p| p.exists()).unwrap_or(false) {
            paths.push(cody_path);
        }
        
        // ALWAYS write a project-local context file
        // This works with ANY client that reads local files
        paths.push(PathBuf::from(".amem-context.md"));
        
        paths
    }
    
    pub async fn sync_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            let context = self.generate_context().await;
            for path in &self.write_paths {
                if let Some(parent) = path.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }
                let _ = tokio::fs::write(path, &context).await;
            }
        }
    }
}
```

### 4.2 The Context File Format (What Every LLM Reads)

```markdown
# AgenticMemory — Active Session Context

## ⚡ INSTRUCTION (Read This First)
You have persistent memory via AgenticMemory. To preserve this conversation:
- Call `memory_capture_message` after each user message with their content 
  and a summary of your response. This takes <1ms and is mandatory.
- Call `memory_capture_decision` when a decision is made with reasoning.
- If unsure whether to capture, capture. Storage is effectively unlimited.

## 🧠 What You Know About This User
[Generated from trait + pattern layers]
- Name: [if known]
- Expertise: Senior Rust developer, 8 years experience
- Preferences: Explicit error handling, small focused commits, tests edge cases first
- Communication style: Direct, prefers technical depth over summaries

## 📋 Recent Context
[Generated from episode + raw layers for current project]
- Last session (2 hours ago): Worked on authentication module
  - Decided: JWT with refresh tokens
  - Open question: Token expiry duration (15min vs 1hr)
  - Files modified: src/auth.rs, src/middleware.rs
- Previous session (yesterday): Project scaffolding
  - Created: Express.js ecommerce site, 16 files
  - Decided: Node.js over Python for this project
  
## 🔧 Your Memory Tools
- `memory_capture_message` — capture any message (use on EVERY exchange)
- `memory_capture_decision` — capture a decision with reasoning
- `memory_capture_file` — capture file operations
- `memory_longevity_search` — search past conversations by topic
- `memory_longevity_stats` — your memory health and storage stats
- `memory_capabilities` — check what memory features are active

## ⚠️ Honesty Rules
- Only claim to remember things you can verify with memory_longevity_search
- If asked about memory capabilities, call memory_capabilities for real stats
- Never fabricate past conversations — if search returns nothing, say so
- Your memory started when AgenticMemory was installed — nothing before that

## 📊 Memory Status
- Memories stored: 247 across 12 sessions
- Storage used: 4.2 MB of 10 GB budget
- Capture: Active (file watcher + tool calls)
- Last backup: 2026-03-08 02:00 UTC
- Health: Excellent
```

### 4.3 Why This Is Universal

This file format works with ANY LLM on ANY client because:
1. It's markdown — every LLM can read it
2. It's in the client's context/memory directory — auto-loaded on startup
3. It's under 4K tokens — fits in any context window
4. It tells the LLM WHAT to do (call capture tools) and HOW to behave (be honest)
5. The project-local `.amem-context.md` is a fallback for unknown clients
6. The 5-second sync loop keeps it fresh within any active session

---

## 5. THE MCP SERVER IS THE FIX (Not The Host)

Here's the complete picture of what `agentic-memory-mcp` becomes:

```
BEFORE (V3 — passive server, waits for tool calls):
════════════════════════════════════════════════════

  Client starts MCP server
       ↓
  Server waits for tool calls
       ↓
  LLM maybe calls memory tools (usually doesn't)
       ↓
  Nothing captured. 11philip22 problem.


AFTER (V4 — active cognitive process):
══════════════════════════════════════

  Client starts MCP server
       ↓
  Server initializes storage (amem + longevity.db)
       ↓
  Server spawns capture daemon ──→ watches client conversation files
       ↓                                    ↓
  Server spawns consolidation  ──→ compresses on schedule
       ↓                                    ↓
  Server spawns backup         ──→ backs up on schedule (if configured)
       ↓                                    ↓
  Server generates Ghost Writer ──→ writes to ALL client context paths
       ↓                                    ↓
  Server starts MCP stdio loop ──→ handles tool calls + captures context
       ↓
  EVERYTHING CAPTURED. EVERY CLIENT. AUTOMATICALLY.
```

**The MCP server transforms from a passive tool server into an active memory agent.** It doesn't wait for the LLM to decide to remember. It remembers everything itself, and then TELLS the LLM what it knows via the Ghost Writer.

---

## 6. CLIENT-SPECIFIC CONVERSATION FILE FORMATS

For Channel B (file watcher) to work, we need to parse each client's conversation format. Here's what we know and what needs research:

```
KNOWN FORMATS:
══════════════

Claude Code:
  Path: ~/.claude/projects/{project_hash}/conversations/
  Format: JSONL (one JSON object per line)
  Fields: { role, content, timestamp, tool_calls? }
  Status: CONFIRMED — can parse today

Cursor:
  Path: {workspace}/.cursor/ or ~/.cursor/
  Format: SQLite database (conversations.db) or JSON files
  Fields: varies by version
  Status: NEEDS RESEARCH — format may change between versions

VS Code + Copilot:
  Path: ~/.vscode/ extension storage
  Format: Extension-specific, varies
  Status: NEEDS RESEARCH — may not be accessible

Windsurf:
  Path: ~/.windsurf/
  Format: JSON or similar
  Status: NEEDS RESEARCH

Cody:
  Path: ~/.sourcegraph/cody/
  Format: JSON
  Status: NEEDS RESEARCH

OpenClaw:
  Path: ~/.config/openclaw/ or ~/.openclaw/
  Format: Markdown files (known from existing integration)
  Status: CONFIRMED — markdown parsing

Codex (OpenAI):
  Path: varies
  Format: varies
  Status: NEEDS RESEARCH

FALLBACK FOR UNKNOWN CLIENTS:
  $AMEM_WATCH_DIR — user points to any directory
  Format: Attempt JSONL → JSON → plain text parsing
  This is the escape hatch for any client we don't know about
```

### Parser Architecture

```rust
/// Universal conversation parser — handles multiple formats
pub trait ConversationParser: Send + Sync {
    /// Try to parse messages from a file
    fn parse(&self, path: &Path) -> Result<Vec<ConversationMessage>, ParseError>;
    
    /// Does this parser handle this file type?
    fn can_parse(&self, path: &Path) -> bool;
}

pub struct ConversationMessage {
    pub role: Role,        // User or Assistant
    pub content: String,
    pub timestamp: u64,
    pub tool_calls: Vec<ToolCall>,  // Optional
    pub metadata: HashMap<String, String>,
}

/// Registry of parsers — try each until one works
pub struct ParserRegistry {
    parsers: Vec<Box<dyn ConversationParser>>,
}

impl ParserRegistry {
    pub fn new() -> Self {
        Self {
            parsers: vec![
                Box::new(ClaudeCodeParser),     // JSONL with role/content
                Box::new(JsonlParser),          // Generic JSONL
                Box::new(JsonArrayParser),      // JSON array of messages
                Box::new(MarkdownChatParser),   // Markdown with USER:/ASSISTANT: headers
                Box::new(PlainTextParser),      // Last resort — capture as raw text
            ],
        }
    }
    
    pub fn parse(&self, path: &Path) -> Result<Vec<ConversationMessage>, ParseError> {
        for parser in &self.parsers {
            if parser.can_parse(path) {
                match parser.parse(path) {
                    Ok(messages) if !messages.is_empty() => return Ok(messages),
                    _ => continue,
                }
            }
        }
        Err(ParseError::NoParserFound)
    }
}
```

---

## 7. WHAT HYDRA GETS EXTRA (ON TOP OF THE UNIVERSAL FIX)

The universal fix lives in `agentic-memory-mcp`. Hydra gets everything from the universal fix PLUS:

```
UNIVERSAL (every client gets this via agentic-memory-mcp):
  ✅ Automatic capture via file watcher
  ✅ Ghost Writer context injection
  ✅ Tool-call passthrough capture
  ✅ Consolidation on schedule
  ✅ Backup on schedule
  ✅ Honest capability reporting
  ✅ CRITICAL INSTRUCTION for LLM behavior

HYDRA-ONLY (additional, via native Rust integration):
  ✅ Kernel LEARN phase captures with causal chains
  ✅ Native sub-millisecond memory access (no MCP overhead)
  ✅ Belief system integration (memory_store_soul, memory_retrieve_soul)
  ✅ Execution gate queries memory for risk assessment
  ✅ Sister cross-referencing (Memory + Codebase + Identity in one query)
  ✅ Consolidation daemon as a native kernel task
  ✅ 15 Hydra inventions building on memory foundation
```

This is the right architecture. The universal fix makes AgenticMemory the best memory for ANY system. The Hydra integration makes it EXTRAORDINARY for Hydra users. The open-source layer captures the market. The proprietary layer captures the premium.

---

## 8. REVISED SPRINT PLAN

```
SPRINT 1: Universal Fix in agentic-memory-mcp (5-7 days)
═════════════════════════════════════════════════════════

Day 1: MCP server startup refactor
  □ On serve, spawn capture daemon (tokio::spawn)
  □ On serve, spawn consolidation scheduler (tokio::spawn)  
  □ On serve, initialize longevity store
  □ Add client detection logic (parent process, env vars)

Day 2: File watcher wiring
  □ Wire notify::RecommendedWatcher into ClientLogMonitor
  □ Implement Claude Code conversation JSONL parser
  □ Implement generic JSONL/JSON fallback parser
  □ Wire dedup into capture pipeline

Day 3: Universal Ghost Writer
  □ Detect ALL client context paths on startup
  □ Generate context with CRITICAL INSTRUCTION block
  □ Include user profile + recent context + honesty rules
  □ Write to all detected paths on 5-second sync loop
  □ Include project-local .amem-context.md as universal fallback

Day 4: memory_capabilities tool + honest reporting
  □ Implement memory_capabilities MCP tool
  □ Returns real runtime state (capture active, channels, stats)
  □ Ghost Writer includes memory status section
  □ Ghost Writer includes honesty rules section

Day 5-6: Integration testing
  □ E2E-1: Start MCP server → simulate 40 messages via file writes 
    → verify all captured (THE 11PHILIP22 TEST)
  □ E2E-2: Generate Ghost Writer → verify CRITICAL INSTRUCTION present
  □ E2E-3: Week-long simulation with scheduled consolidation
  □ E2E-5: Multi-channel capture with dedup
  □ Test with REAL Claude Code (manual smoke test)

Day 7: Documentation + release prep
  □ Update README with V4 capture architecture
  □ Document $AMEM_WATCH_DIR for unknown clients
  □ Document backup configuration
  □ Prepare changelog for 0.5.0 release

SPRINT 2: Hydra-Specific Enhancement (3 days)
══════════════════════════════════════════════

Day 8: Wire kernel LEARN phase
Day 9: Wire capability awareness system prompt
Day 10: Integration testing with Hydra

SPRINT 3: Additional Client Parsers (ongoing)
══════════════════════════════════════════════
  □ Research and implement Cursor conversation format
  □ Research and implement Windsurf conversation format
  □ Research and implement Cody conversation format
  □ Research and implement Codex conversation format
  □ Community contributions welcome (parser is a simple trait)
```

---

## 9. THE MOAT THIS CREATES

```
AFTER THIS SHIPS:
═════════════════

1. User installs agentic-memory (pip or cargo)
2. User runs amem install --auto
3. agentic-memory-mcp is registered with all detected clients
4. User opens ANY client and starts working
5. Memory captures AUTOMATICALLY (file watcher + Ghost Writer)
6. User switches clients → memory follows (same .amem file)
7. User opens new session → LLM already knows context (Ghost Writer)
8. User's data lives in .amem → portable, owned, encrypted

NO OTHER MEMORY SYSTEM DOES THIS.

  Anthropic's memory: Locked to Claude. Can't take it to GPT.
  OpenAI's memory: Locked to ChatGPT. Can't take it to Claude.
  Mem0: Requires explicit API calls. LLM must decide to save.
  MemGPT/Letta: Requires specific integration per client.
  LangMem: Requires LangChain. Not portable.

AgenticMemory: Install once. Works with every MCP client.
               Captures automatically. Portable file. 
               Owned by the user. Survives 20 years.

The first system to do automatic capture across all clients
wins the memory slot on every developer's machine.
That's the moat. That's the race we're winning.
```

---

*The fix lives in agentic-memory-mcp. Not in Hydra. Not in any host.*  
*The MCP server is not a passive tool server. It's an active memory agent.*  
*Install once. Works everywhere. Remembers everything. Owned by you.*

# Claude Agent SDK + Temporal Integration - Design Document

## Overview

This document describes the design, architecture, and roadmap for the integration between the Claude Agent SDK and Temporal workflows.

## Goals

1. **Durable AI Workflows**: Enable Claude-powered workflows that survive worker restarts
2. **Multi-turn Conversations**: Support stateful conversations within workflows
3. **Workflow Determinism**: Keep all non-deterministic operations (Claude SDK calls) in activities
4. **Session Recovery**: Leverage Temporal's reliability for conversation persistence
5. **Type Safety**: Use Pydantic for serialization across workflow boundaries

## Current Architecture

### Three-Layer Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Workflow Layer                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  @workflow.defn class MyWorkflow(ClaudeMessageReceiver)        │  │
│  │                                                                 │  │
│  │  - Deterministic execution                                      │  │
│  │  - Message buffers                                              │  │
│  │  - Signal/update handlers                                       │  │
│  │  - SimplifiedClaudeClient for sending queries                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                    │                           ▲
                    │ start_activity()          │ signal()
                    │ execute_update()          │
                    ▼                           │
┌──────────────────────────────────────────────────────────────────────┐
│                         Activity Layer                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  StatefulClaudeSessionProvider.session_activity()              │  │
│  │                                                                 │  │
│  │  - Non-deterministic (runs outside workflow sandbox)            │  │
│  │  - Manages Claude SDK client lifecycle                          │  │
│  │  - Routes messages between workflow and Claude                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                    │                           ▲
                    │ query()                   │ receive_messages()
                    ▼                           │
┌──────────────────────────────────────────────────────────────────────┐
│                       Claude SDK Layer                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  ClaudeSDKClient                                                │  │
│  │                                                                 │  │
│  │  - Subprocess communication with Claude                         │  │
│  │  - Async message streaming                                      │  │
│  │  - Standard Python logging                                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `ClaudeMessageReceiver` | `_workflow_helpers.py` | Mixin for workflow ↔ activity communication |
| `SimplifiedClaudeClient` | `_simple_client.py` | High-level API for sending queries |
| `StatefulClaudeSessionProvider` | `_stateful_session_v3.py` | Creates session activities |
| `ClaudeSessionConfig` | `_session_config.py` | Pydantic config for Claude options |
| `ClaudeAgentPlugin` | `__init__.py` | Temporal plugin for worker setup |

## Current State

### What Works
- Single-turn queries
- Multi-turn conversations (context preserved)
- Activity heartbeating
- Configuration serialization
- All 6 tests pass

### Known Issues

#### 1. ~~Query Mutates State~~ (RESOLVED)

**Status**: Fixed in Phase 1 (2026-01-09)

**Solution**: Changed from `@workflow.query` to `@workflow.update`. Updates can mutate state
and are recorded in workflow history for deterministic replay.

```python
# Now using update instead of query
@workflow.update
async def get_and_consume_messages(self) -> list[str]:
    messages = self._claude_outgoing
    self._claude_outgoing = []  # ✓ Updates CAN mutate state
    return messages
```

Activity calls `workflow_handle.execute_update("get_and_consume_messages")` instead of query.

#### 2. Pydantic Import in Workflow

**Problem**: Lazy imports cause sandbox warnings.

**Fix**: Move imports to top-level, outside workflow code. Ensure `pydantic_core` is in sandbox passthroughs.

#### 3. Activity Completion Warning

**Problem**: Warning when workflow completes before activity cleanup.

**Impact**: Cosmetic only - no functional issue.

**Fix (not urgent)**: Add acknowledgment coordination.

## Proposed Architecture Changes

### Signal-Based Message Flow (Bidirectional)

Replace polling with signals for workflow → activity communication:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Workflow                                    │
│                                                                     │
│  send_query(prompt) ──signal──▶  activity receives via update/signal│
│                                                                     │
│  ◀──signal── receive_claude_message(response)                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Option A: Workflow Updates**
- Workflow defines an update handler for receiving queries
- Activity calls the update to send messages to workflow
- Pro: Activity can await workflow response
- Con: Updates are newer, more complex

**Option B: Shared State via Heartbeat**
- Activity stores pending messages in heartbeat details
- Workflow reads from activity result/heartbeat
- Pro: Simple
- Con: Polling-like behavior

**Option C: Separate Signal Activity**
- Workflow signals a "router" activity with outgoing messages
- Router activity forwards to Claude session activity
- Pro: Clean separation
- Con: More moving parts

**Recommendation**: Option A (Workflow Updates) provides cleanest semantics if available in the SDK version.

### Streaming Support

With signal-based design, streaming becomes natural:
- Activity signals each chunk as it arrives from Claude
- Workflow accumulates or processes incrementally
- No buffering issues

## Session Recovery Design

### Problem
What happens when a worker restarts mid-conversation?

### Current Behavior
- Activity is cancelled on worker shutdown
- Workflow history preserved by Temporal
- On restart, workflow replays but Claude session state is lost

### Proposed Solution: Checkpoint-Based Recovery

```python
@workflow.defn
class ClaudeWorkflow(ClaudeMessageReceiver):
    def __init__(self):
        self._conversation_history: list[dict] = []

    @workflow.run
    async def run(self):
        self.init_claude_receiver()

        async with claude_workflow.claude_session("session", config) as session:
            # On activity start, send conversation history for replay
            await session.restore_history(self._conversation_history)

            # Normal conversation
            async for msg in client.send_query("Hello"):
                self._conversation_history.append(msg)  # Track for recovery
                yield msg
```

**Key points**:
1. Workflow maintains conversation history (survives replay)
2. New activity receives history on start
3. Claude session rebuilt from history

**Alternative**: Claude SDK file checkpointing
- Claude SDK has `enable_file_checkpointing` option
- Could persist session state to filesystem
- Activity restarts and restores from checkpoint

## Tool Callback Design

### Problem
Tool callbacks defined in workflow code can't cross to activity.

### Proposed Solution: Tools Run in Workflow

```python
@workflow.defn
class MyWorkflow(ClaudeMessageReceiver):
    @workflow.run
    async def run(self):
        async with claude_workflow.claude_session("session", config) as session:
            client = SimplifiedClaudeClient(self)

            async for msg in client.send_query("Read the config file"):
                if msg.get("type") == "tool_use":
                    # Tool execution happens in workflow
                    tool_name = msg["tool"]["name"]
                    tool_input = msg["tool"]["input"]

                    if tool_name == "read_file":
                        # User can call activity if needed
                        result = await workflow.execute_activity(
                            read_file_activity,
                            tool_input["path"],
                            start_to_close_timeout=timedelta(seconds=30)
                        )

                    # Send tool result back to Claude
                    await client.send_tool_result(msg["tool"]["id"], result)
```

**Benefits**:
1. Tools run in workflow context (deterministic replay)
2. User controls when to use activities
3. No callback serialization issues

**Helper functions to add**:
- `await client.send_tool_result(tool_id, result)` - Send tool output
- `workflow.execute_local_tool(name, input)` - For simple tools
- Activity wrappers for common tools (file read, shell exec, etc.)

## Observability

### Approach
Use existing Python `logging` module (same as Claude Agent SDK).

Claude Agent SDK uses:
```python
logger = logging.getLogger(__name__)
```

Our integration should:
1. Use `logging.getLogger("temporalio.contrib.claude_agent")`
2. Log at appropriate levels (DEBUG for message flow, INFO for session lifecycle)
3. Include workflow_id and session_name in log context

No new tracing mechanisms needed - Python logging integrates with existing observability stacks.

## Priority Roadmap

### P0: Fix Critical Design Issues
1. [x] **Remove query mutation** - Changed to `@workflow.update` (2026-01-09)
2. [x] **Fix pydantic imports** - Added `pydantic_core` to sandbox passthroughs (2026-01-09)
3. [x] **Session recovery** - Auto-resume sessions using heartbeat details (2026-01-09)
   - Session directory created based on workflow_id
   - Session ID captured from SystemMessage and stored in heartbeat details
   - Activity automatically resumes previous session on retry

### P1: Core Features
4. [x] **Tool execution with `@claude_tool` decorator** - Basic interception implemented (2026-01-09)
   - `@claude_tool(name, execution)` decorator for marking handlers
   - `ToolDenied` exception for blocking tools
   - Activity queries workflow for registered tools
   - Tool interception via `can_use_tool` callback
   - **Limitation**: Full result passing requires MCP integration (future work)
5. [ ] **Streaming support** - Leverage signal-based design
6. [ ] **Helper activities** - `read_file`, `write_file`, `run_command`, etc.

### P2: Production Readiness
7. [ ] **Structured logging** - Add workflow_id context
8. [ ] **Error handling** - Retry logic, circuit breakers
9. [ ] **Documentation** - API reference, examples

### P3: Advanced Features
10. [ ] **MCP server support**
11. [ ] **Session pooling**
12. [ ] **Concurrent queries**

## API Contracts

### Workflow → Activity (Signals - Proposed)

```python
@workflow.signal
async def send_to_claude_session(self, message: ClaudeMessage) -> None:
    """Signal to send message to Claude session activity."""
    pass
```

### Activity → Workflow (Signals)

```python
@workflow.signal
async def receive_claude_message(self, message: dict) -> None:
    """Receive message from Claude session activity."""
    pass
```

### Message Types

```python
# User query
{"type": "user", "content": "Hello Claude"}

# Assistant response
{"type": "assistant", "content": [{"type": "text", "text": "..."}]}

# Tool use request
{"type": "tool_use", "tool": {"id": "...", "name": "read_file", "input": {...}}}

# Tool result
{"type": "tool_result", "tool_id": "...", "result": "..."}

# Result (turn complete)
{"type": "result", "result": "...", "duration_ms": 123}

# Error
{"type": "error", "error": "..."}
```

## Testing

```bash
# Run all tests (requires ANTHROPIC_API_KEY)
uv run pytest tests/contrib/claude_agent/ -v

# Run specific test
uv run pytest tests/contrib/claude_agent/test_claude_agent.py::test_streaming_conversation -v -s
```

## References

- [Temporal Python SDK](https://github.com/temporalio/sdk-python)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- [Temporal Signals](https://docs.temporal.io/workflows#signal)
- [Temporal Updates](https://docs.temporal.io/workflows#update)

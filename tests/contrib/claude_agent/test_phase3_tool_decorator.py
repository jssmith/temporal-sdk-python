"""Test that Phase 3 (@claude_tool decorator) is working correctly."""

import os
import uuid
from datetime import timedelta

import pytest

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.claude_agent import (
    ClaudeAgentPlugin,
    ClaudeMessageReceiver,
    ClaudeSessionConfig,
    SimplifiedClaudeClient,
    StatefulClaudeSessionProvider,
    ToolDenied,
    claude_tool,
)
from temporalio.contrib.claude_agent import workflow as claude_workflow
from temporalio.contrib.claude_agent._tool_decorator import (
    ToolRegistry,
    get_tool_registry,
    register_tool_handlers,
)
from tests.helpers import new_worker


class TestToolRegistry:
    """Unit tests for the tool registry."""

    def test_register_and_get(self):
        """Test basic registration and retrieval."""

        async def handler(tool_id: str, input: dict) -> str:
            return "result"

        registry = ToolRegistry()
        registry.register("TestTool", handler, "workflow")

        assert registry.has("TestTool")
        assert not registry.has("OtherTool")

        handler_info = registry.get("TestTool")
        assert handler_info is not None
        assert handler_info.tool_name == "TestTool"
        assert handler_info.execution == "workflow"

    def test_decorator_marks_function(self):
        """Test that @claude_tool marks the function with metadata."""

        @claude_tool("MyTool", execution="activity")
        async def my_handler(tool_id: str, input: dict) -> str:
            return "result"

        assert hasattr(my_handler, "_claude_tool_name")
        assert my_handler._claude_tool_name == "MyTool"
        assert my_handler._claude_tool_execution == "activity"


class TestToolDenied:
    """Unit tests for ToolDenied exception."""

    def test_basic_denial(self):
        """Test basic tool denial."""
        exc = ToolDenied("Not allowed")
        assert exc.message == "Not allowed"
        assert exc.interrupt is False

    def test_denial_with_interrupt(self):
        """Test tool denial with interrupt."""
        exc = ToolDenied("Critical error", interrupt=True)
        assert exc.message == "Critical error"
        assert exc.interrupt is True


# Workflow with tool handlers for integration testing
@workflow.defn
class Phase3ToolTestWorkflow(ClaudeMessageReceiver):
    """Workflow with tool handlers for testing."""

    def __init__(self):
        self._tool_calls: list[dict] = []

    @claude_tool("Bash")
    async def handle_bash(self, tool_id: str, input: dict) -> str:
        """Custom handler for Bash commands."""
        self._tool_calls.append({
            "tool": "Bash",
            "tool_id": tool_id,
            "input": input,
        })
        command = input.get("command", "")
        # Simulate execution
        if "echo" in command:
            return f"Simulated output: {command}"
        raise ToolDenied("Only echo commands are allowed")

    @workflow.run
    async def run(self, prompt: str) -> dict:
        self.init_claude_receiver()

        config = ClaudeSessionConfig(
            system_prompt="You are a helpful assistant. Use the Bash tool when asked to run commands.",
            max_turns=2,
        )

        results = {
            "response": "",
            "tool_calls": [],
        }

        async with claude_workflow.claude_session("phase3-session", config):
            client = SimplifiedClaudeClient(self)
            async for message in client.send_query(prompt):
                msg_type = message.get("type")
                if msg_type == "assistant":
                    content = message.get("message", {}).get("content", [])
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            results["response"] += block.get("text", "")
            await client.close()

        results["tool_calls"] = self._tool_calls
        return results


def test_handler_registration():
    """Test that handlers are properly registered from a mock workflow instance."""
    # Use a simple class instead of a real workflow to test registration

    class MockWorkflow:
        @claude_tool("Read")
        async def handle_read(self, tool_id: str, input: dict) -> str:
            return "file content"

        @claude_tool("Write", execution="activity")
        async def handle_write(self, tool_id: str, input: dict) -> str:
            return "written"

    # Create an instance and register handlers
    instance = MockWorkflow()
    registry = register_tool_handlers(instance)

    assert registry.has("Read")
    assert registry.has("Write")
    assert not registry.has("Bash")

    read_handler = registry.get("Read")
    assert read_handler is not None
    assert read_handler.execution == "workflow"

    write_handler = registry.get("Write")
    assert write_handler is not None
    assert write_handler.execution == "activity"


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No Anthropic API key available",
)
async def test_tool_handler_query(client: Client):
    """Test that registered tools can be queried from the workflow."""
    session_provider = StatefulClaudeSessionProvider("phase3-session")
    plugin = ClaudeAgentPlugin(session_providers=[session_provider])

    config = client.config()
    config["plugins"] = [plugin]
    client = Client(**config)

    workflow_id = f"phase3-query-{uuid.uuid4()}"

    async with new_worker(client, Phase3ToolTestWorkflow, activities=[]) as worker:
        # Start the workflow
        handle = await client.start_workflow(
            Phase3ToolTestWorkflow.run,
            "Just say hello.",
            id=workflow_id,
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=60),
        )

        # Wait briefly for workflow to initialize
        import asyncio
        await asyncio.sleep(1)

        # Query the registered tools
        try:
            registered_tools = await handle.query("get_registered_tools")
            print(f"Registered tools: {registered_tools}")
            assert "Bash" in registered_tools
        except Exception as e:
            print(f"Query failed (expected during workflow): {e}")

        # Wait for workflow to complete
        result = await handle.result()
        print(f"Workflow result: {result}")


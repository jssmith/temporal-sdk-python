"""Tool execution decorator for Claude Agent workflows.

This module provides the @claude_tool decorator for intercepting and handling
Claude tool execution in Temporal workflows.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal


class ToolDenied(Exception):
    """Exception raised when a tool execution is denied.

    Raise this in a @claude_tool handler to block the tool and send
    an error message back to Claude.

    Example:
        @claude_tool("Bash")
        async def handle_bash(self, tool_id: str, input: dict) -> str:
            if "rm -rf" in input.get("command", ""):
                raise ToolDenied("Dangerous command blocked")
            return await workflow.execute_activity(...)
    """

    def __init__(self, message: str, interrupt: bool = False):
        """Initialize the exception.

        Args:
            message: Error message to send to Claude
            interrupt: If True, interrupt the entire session
        """
        super().__init__(message)
        self.message = message
        self.interrupt = interrupt


@dataclass
class ToolHandler:
    """Metadata for a registered tool handler."""

    tool_name: str
    execution: Literal["claude", "activity", "workflow"]
    handler: Callable[[Any, str, dict], Awaitable[str]]


@dataclass
class ToolRegistry:
    """Registry for tool handlers in a workflow.

    This is used internally to track which tools have handlers registered
    and how they should be executed.
    """

    handlers: dict[str, ToolHandler] = field(default_factory=dict)

    def register(
        self,
        tool_name: str,
        handler: Callable[[Any, str, dict], Awaitable[str]],
        execution: Literal["claude", "activity", "workflow"],
    ) -> None:
        """Register a tool handler.

        Args:
            tool_name: Name of the tool (e.g., "Read", "Write", "Bash")
            handler: Async function to handle the tool
            execution: Where the tool runs
        """
        self.handlers[tool_name] = ToolHandler(
            tool_name=tool_name,
            execution=execution,
            handler=handler,
        )

    def get(self, tool_name: str) -> ToolHandler | None:
        """Get a handler for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolHandler if registered, None otherwise
        """
        return self.handlers.get(tool_name)

    def has(self, tool_name: str) -> bool:
        """Check if a tool has a handler registered.

        Args:
            tool_name: Name of the tool

        Returns:
            True if the tool has a handler
        """
        return tool_name in self.handlers


# Global registry is set per-workflow instance
_WORKFLOW_TOOL_REGISTRY: dict[int, ToolRegistry] = {}


def get_tool_registry(workflow_instance: Any) -> ToolRegistry:
    """Get or create a tool registry for a workflow instance.

    Args:
        workflow_instance: The workflow instance

    Returns:
        ToolRegistry for this workflow
    """
    instance_id = id(workflow_instance)
    if instance_id not in _WORKFLOW_TOOL_REGISTRY:
        _WORKFLOW_TOOL_REGISTRY[instance_id] = ToolRegistry()
    return _WORKFLOW_TOOL_REGISTRY[instance_id]


def clear_tool_registry(workflow_instance: Any) -> None:
    """Clear the tool registry for a workflow instance.

    Called during workflow cleanup.

    Args:
        workflow_instance: The workflow instance
    """
    instance_id = id(workflow_instance)
    if instance_id in _WORKFLOW_TOOL_REGISTRY:
        del _WORKFLOW_TOOL_REGISTRY[instance_id]


def claude_tool(
    tool_name: str,
    execution: Literal["claude", "activity", "workflow"] = "workflow",
):
    """Decorator to register a handler for a Claude tool.

    Use this decorator on workflow methods to intercept and handle specific
    Claude tools. The handler will be called when Claude attempts to use
    the specified tool.

    Args:
        tool_name: Name of the tool to handle (e.g., "Read", "Write", "Bash")
        execution: Where the tool runs:
            - "claude": Let Claude execute normally (no interception)
            - "activity": Run handler, use activity for durability
            - "workflow": Run handler in workflow context

    Returns:
        Decorator function

    Example:
        @workflow.defn
        class MyWorkflow(ClaudeMessageReceiver):

            @claude_tool("Write", execution="activity")
            async def handle_write(self, tool_id: str, input: dict) -> str:
                await workflow.execute_activity(
                    write_file_activity,
                    input["file_path"],
                    input["content"],
                    start_to_close_timeout=timedelta(seconds=30)
                )
                return "File written successfully"

            @claude_tool("Bash")
            async def handle_bash(self, tool_id: str, input: dict) -> str:
                command = input.get("command", "")
                if "rm -rf" in command:
                    raise ToolDenied("Dangerous command blocked")
                return await workflow.execute_activity(
                    run_command_activity,
                    command,
                    start_to_close_timeout=timedelta(minutes=5)
                )
    """

    def decorator(func: Callable[[Any, str, dict], Awaitable[str]]):
        # Mark the function with tool metadata
        func._claude_tool_name = tool_name
        func._claude_tool_execution = execution
        return func

    return decorator


def register_tool_handlers(workflow_instance: Any) -> ToolRegistry:
    """Scan a workflow instance for @claude_tool handlers and register them.

    This is called automatically when init_claude_receiver() is called.

    Args:
        workflow_instance: The workflow instance to scan

    Returns:
        ToolRegistry with all registered handlers
    """
    registry = get_tool_registry(workflow_instance)

    # Scan for methods with _claude_tool_name attribute
    for attr_name in dir(workflow_instance):
        try:
            attr = getattr(workflow_instance, attr_name)
            if callable(attr) and hasattr(attr, "_claude_tool_name"):
                tool_name = attr._claude_tool_name
                execution = attr._claude_tool_execution
                registry.register(tool_name, attr, execution)
        except Exception:
            # Skip attributes that can't be accessed
            pass

    return registry


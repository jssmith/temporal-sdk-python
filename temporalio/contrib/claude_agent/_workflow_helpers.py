"""Helper classes and mixins for workflows using Claude Agent SDK."""

import asyncio
from typing import Any

from pydantic import BaseModel
from temporalio import workflow

from ._tool_decorator import (
    ToolDenied,
    ToolRegistry,
    clear_tool_registry,
    get_tool_registry,
    register_tool_handlers,
)


class ToolRequest(BaseModel):
    """A request to execute a tool in the workflow."""

    tool_id: str
    tool_name: str
    input: dict[str, Any]


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_id: str
    success: bool
    result: str | None = None
    error: str | None = None
    interrupt: bool = False


class ClaudeMessageReceiver:
    """Mixin for workflows that need to communicate with Claude sessions.

    This class provides signal handlers, query methods, and message buffers needed
    for bidirectional communication with Claude sessions. Workflows should inherit
    from this class to gain the ability to send and receive Claude messages.

    Example:
        >>> from temporalio import workflow
        >>> from temporalio.contrib.claude_agent import ClaudeMessageReceiver
        >>>
        >>> @workflow.defn
        >>> class MyWorkflow(ClaudeMessageReceiver):
        ...     @workflow.run
        ...     async def run(self) -> str:
        ...         # Initialize the receiver
        ...         self.init_claude_receiver()
        ...         # Send messages via self.send_to_claude()
        ...         # Receive messages via signals and wait_for_claude_messages()
    """

    def init_claude_receiver(self) -> None:
        """Initialize the Claude message receiver.

        Must be called in the workflow's run method before using Claude sessions.
        Also registers any @claude_tool decorated handlers.
        """
        self._claude_messages: list[dict[str, Any]] = []
        self._claude_outgoing: list[str] = []
        self._claude_message_event = asyncio.Event()

        # Tool execution state
        self._pending_tool_requests: list[ToolRequest] = []
        self._tool_results: dict[str, ToolResult] = {}
        self._tool_request_event = asyncio.Event()
        self._tool_result_events: dict[str, asyncio.Event] = {}

        # Register tool handlers from @claude_tool decorators
        self._tool_registry = register_tool_handlers(self)

    @workflow.signal
    async def receive_claude_message(self, message: dict[str, Any]) -> None:
        """Receive a message from the Claude session activity.

        This signal handler is called by the session activity when messages
        are received from Claude.

        Args:
            message: The message received from Claude
        """
        if not hasattr(self, "_claude_messages"):
            # Initialize if not already done
            self.init_claude_receiver()

        workflow.logger.debug(f"Received Claude message: {message.get('type')}")
        self._claude_messages.append(message)
        self._claude_message_event.set()

    async def wait_for_claude_messages(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Wait for and retrieve Claude messages.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            List of messages received (may be empty if timeout)
        """
        if not hasattr(self, "_claude_messages"):
            self.init_claude_receiver()

        if timeout is not None:
            try:
                await asyncio.wait_for(self._claude_message_event.wait(), timeout)
            except asyncio.TimeoutError:
                pass
        else:
            await self._claude_message_event.wait()

        # Get and clear messages
        messages = self._claude_messages
        self._claude_messages = []
        self._claude_message_event.clear()
        return messages

    def get_claude_messages(self) -> list[dict[str, Any]]:
        """Get any buffered Claude messages without waiting.

        Returns:
            List of buffered messages (may be empty)
        """
        if not hasattr(self, "_claude_messages"):
            self.init_claude_receiver()

        messages = self._claude_messages
        self._claude_messages = []
        self._claude_message_event.clear()
        return messages

    @workflow.update
    async def get_and_consume_messages(self) -> list[str]:
        """Get and consume outgoing messages for Claude (called by session activity).

        This is an update (not a query) because it mutates workflow state by clearing
        the message buffer. Updates are recorded in workflow history and replay
        deterministically.

        Returns:
            List of messages to send to Claude
        """
        if not hasattr(self, "_claude_outgoing"):
            self.init_claude_receiver()

        messages = self._claude_outgoing
        self._claude_outgoing = []
        return messages

    def send_to_claude(self, message: str) -> None:
        """Queue a message to send to Claude.

        Args:
            message: Raw message to send (typically JSON + newline)
        """
        if not hasattr(self, "_claude_outgoing"):
            self.init_claude_receiver()

        self._claude_outgoing.append(message)

    # ========== Tool Execution Support ==========

    def get_tool_registry(self) -> ToolRegistry:
        """Get the tool registry for this workflow.

        Returns:
            ToolRegistry containing registered handlers
        """
        if not hasattr(self, "_tool_registry"):
            self._tool_registry = register_tool_handlers(self)
        return self._tool_registry

    @workflow.signal
    async def request_tool_execution(self, request_dict: dict[str, Any]) -> None:
        """Signal from activity requesting tool execution.

        Args:
            request_dict: Tool request as a dictionary
        """
        if not hasattr(self, "_pending_tool_requests"):
            self.init_claude_receiver()

        request = ToolRequest.model_validate(request_dict)
        workflow.logger.debug(f"Received tool request: {request.tool_name}")

        # Create result event for this tool
        self._tool_result_events[request.tool_id] = asyncio.Event()

        # Execute the tool handler
        registry = self.get_tool_registry()
        handler = registry.get(request.tool_name)

        if handler is None:
            # No handler registered - this shouldn't happen as activity checks first
            result = ToolResult(
                tool_id=request.tool_id,
                success=False,
                error=f"No handler registered for tool: {request.tool_name}",
            )
        else:
            try:
                # Execute the handler
                output = await handler.handler(request.tool_id, request.input)
                result = ToolResult(
                    tool_id=request.tool_id,
                    success=True,
                    result=output,
                )
            except ToolDenied as e:
                result = ToolResult(
                    tool_id=request.tool_id,
                    success=False,
                    error=e.message,
                    interrupt=e.interrupt,
                )
            except Exception as e:
                result = ToolResult(
                    tool_id=request.tool_id,
                    success=False,
                    error=str(e),
                )

        # Store result and signal completion
        self._tool_results[request.tool_id] = result
        self._tool_result_events[request.tool_id].set()

    @workflow.update
    async def get_tool_result(self, tool_id: str) -> dict[str, Any]:
        """Update to get tool result (called by session activity).

        Waits for the tool to complete and returns the result.

        Args:
            tool_id: ID of the tool to get result for

        Returns:
            ToolResult as dictionary
        """
        if not hasattr(self, "_tool_results"):
            self.init_claude_receiver()

        # Wait for result if not ready
        if tool_id in self._tool_result_events:
            await self._tool_result_events[tool_id].wait()

        # Get and remove the result
        result = self._tool_results.pop(tool_id, None)
        if tool_id in self._tool_result_events:
            del self._tool_result_events[tool_id]

        if result is None:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                error="Tool result not found",
            ).model_dump()

        return result.model_dump()

    @workflow.query
    def get_registered_tools(self) -> list[str]:
        """Query to get list of registered tool names.

        Returns:
            List of tool names that have handlers registered
        """
        registry = self.get_tool_registry()
        return list(registry.handlers.keys())
"""Test that Phase 1 (query->update) is working correctly."""

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
)
from temporalio.contrib.claude_agent import workflow as claude_workflow
from tests.helpers import new_worker


@workflow.defn
class Phase1VerificationWorkflow(ClaudeMessageReceiver):
    """Workflow to verify Phase 1 update mechanism works correctly."""

    @workflow.run
    async def run(self, prompt: str) -> str:
        self.init_claude_receiver()

        config = ClaudeSessionConfig(
            system_prompt="You are a helpful assistant. Answer concisely.",
            max_turns=1,
        )

        async with claude_workflow.claude_session("phase1-session", config):
            client = SimplifiedClaudeClient(self)
            result = ""
            async for message in client.send_query(prompt):
                if message.get("type") == "assistant":
                    content = message.get("message", {}).get("content", [])
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            result += block.get("text", "")
            await client.close()
            return result


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No Anthropic API key available",
)
async def test_update_appears_in_history(client: Client):
    """Verify that get_and_consume_messages update appears in workflow history."""
    session_provider = StatefulClaudeSessionProvider("phase1-session")
    plugin = ClaudeAgentPlugin(session_providers=[session_provider])

    config = client.config()
    config["plugins"] = [plugin]
    client = Client(**config)

    workflow_id = f"phase1-verification-{uuid.uuid4()}"

    async with new_worker(client, Phase1VerificationWorkflow, activities=[]) as worker:
        # Run workflow
        result = await client.execute_workflow(
            Phase1VerificationWorkflow.run,
            "What is 2 + 2? Just give me the number.",
            id=workflow_id,
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=60),
        )

        # Verify we got a result
        assert "4" in result, f"Expected '4' in result, got: {result}"
        print(f"Result: {result}")

        # Fetch workflow history and check for update events
        handle = client.get_workflow_handle(workflow_id)
        history = await handle.fetch_history()

        from temporalio.api.enums.v1 import EventType

        update_events = []
        all_events = []
        for event in history.events:
            # event_type is an int, use EventType.Name() to get the string name
            event_type_int = event.event_type
            event_type_name = EventType.Name(event_type_int)
            all_events.append(event_type_name)
            if "UPDATE" in event_type_name:
                update_events.append(event_type_name)

        print(f"All event types: {all_events}")
        print(f"Update events found: {update_events}")

        # Verify update events are present
        assert len(update_events) > 0, (
            f"No UPDATE events found in workflow history. "
            f"All events: {all_events}"
        )

        print("✓ Phase 1 verified: Updates are being recorded in workflow history")

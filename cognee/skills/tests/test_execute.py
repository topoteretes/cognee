"""Tests for cognee.skills.execute — unit tests with mocked LLM."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock


def _make_mock_response(content: str):
    """Build a mock litellm response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


SAMPLE_SKILL = {
    "skill_id": "summarize",
    "name": "Summarize",
    "instructions": "Condense the input into 2-3 key bullet points.",
    "instruction_summary": "Summarizes text into bullet points.",
    "description": "A summarization skill.",
    "tags": ["context-management"],
    "complexity": "simple",
    "source_path": "",
    "task_patterns": [],
}


class TestExecuteSkill(unittest.TestCase):
    @patch("cognee.skills.execute.litellm.acompletion")
    @patch("cognee.skills.execute.get_llm_config")
    def test_execute_returns_output(self, mock_config, mock_acompletion):
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini", llm_api_key="test")
        mock_acompletion.return_value = _make_mock_response("- Point 1\n- Point 2")

        from cognee.skills.execute import execute_skill

        result = asyncio.run(
            execute_skill(skill=SAMPLE_SKILL, task_text="Summarize this article about AI")
        )

        assert result["success"] is True
        assert result["output"] == "- Point 1\n- Point 2"
        assert result["skill_id"] == "summarize"
        assert result["model"] == "openai/gpt-4o-mini"
        assert result["latency_ms"] >= 0
        assert result["error"] is None

        # Verify the LLM was called with correct structure
        call_args = mock_acompletion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Summarize" in messages[0]["content"]
        assert "Condense the input" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "Summarize this article about AI" in messages[1]["content"]

    @patch("cognee.skills.execute.litellm.acompletion")
    @patch("cognee.skills.execute.get_llm_config")
    def test_execute_with_context(self, mock_config, mock_acompletion):
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini", llm_api_key="test")
        mock_acompletion.return_value = _make_mock_response("Done")

        from cognee.skills.execute import execute_skill

        result = asyncio.run(
            execute_skill(
                skill=SAMPLE_SKILL,
                task_text="Summarize this",
                context="The article is about quantum computing.",
            )
        )

        assert result["success"] is True
        call_args = mock_acompletion.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "quantum computing" in user_msg

    @patch("cognee.skills.execute.litellm.acompletion")
    @patch("cognee.skills.execute.get_llm_config")
    def test_execute_handles_llm_error(self, mock_config, mock_acompletion):
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini", llm_api_key="test")
        mock_acompletion.side_effect = Exception("API rate limit exceeded")

        from cognee.skills.execute import execute_skill

        result = asyncio.run(
            execute_skill(skill=SAMPLE_SKILL, task_text="Summarize this")
        )

        assert result["success"] is False
        assert result["output"] == ""
        assert "rate limit" in result["error"]
        assert result["latency_ms"] >= 0

    def test_client_execute_skill_not_found(self):
        """Client.execute returns error when skill doesn't exist."""
        from cognee.skills.client import Skills

        client = Skills()

        async def _run():
            with patch.object(client, "load", new_callable=AsyncMock, return_value=None):
                return await client.execute("nonexistent", "do something")

        result = asyncio.run(_run())

        assert result["success"] is False
        assert "not found" in result["error"]

    @patch("cognee.skills.execute.litellm.acompletion")
    @patch("cognee.skills.execute.get_llm_config")
    def test_client_execute_auto_observe(self, mock_config, mock_acompletion):
        """Client.execute auto-records the run when auto_observe=True."""
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini", llm_api_key="test")
        mock_acompletion.return_value = _make_mock_response("Result text")

        from cognee.skills.client import Skills

        client = Skills()

        async def _run():
            with patch.object(client, "load", new_callable=AsyncMock, return_value=SAMPLE_SKILL):
                with patch.object(client, "observe", new_callable=AsyncMock) as mock_observe:
                    result = await client.execute("summarize", "do something")
                    return result, mock_observe

        result, mock_observe = asyncio.run(_run())

        assert result["success"] is True
        mock_observe.assert_called_once()
        obs_args = mock_observe.call_args[0][0]
        assert obs_args["selected_skill_id"] == "summarize"
        assert obs_args["success_score"] == 1.0

    @patch("cognee.skills.execute.litellm.acompletion")
    @patch("cognee.skills.execute.get_llm_config")
    def test_client_execute_no_observe(self, mock_config, mock_acompletion):
        """Client.execute skips observation when auto_observe=False."""
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini", llm_api_key="test")
        mock_acompletion.return_value = _make_mock_response("Result")

        from cognee.skills.client import Skills

        client = Skills()

        async def _run():
            with patch.object(client, "load", new_callable=AsyncMock, return_value=SAMPLE_SKILL):
                with patch.object(client, "observe", new_callable=AsyncMock) as mock_observe:
                    result = await client.execute(
                        "summarize", "do something", auto_observe=False
                    )
                    return result, mock_observe

        result, mock_observe = asyncio.run(_run())

        assert result["success"] is True
        mock_observe.assert_not_called()


if __name__ == "__main__":
    unittest.main()

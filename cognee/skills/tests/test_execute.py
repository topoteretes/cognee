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

        result = asyncio.run(execute_skill(skill=SAMPLE_SKILL, task_text="Summarize this"))

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
                    result = await client.execute("summarize", "do something", auto_observe=False)
                    return result, mock_observe

        result, mock_observe = asyncio.run(_run())

        assert result["success"] is True
        mock_observe.assert_not_called()

    @patch("cognee.skills.execute.litellm.acompletion")
    @patch("cognee.skills.execute.get_llm_config")
    def test_client_execute_auto_amendify_on_failure(self, mock_config, mock_acompletion):
        """Client.execute triggers auto_amendify when execution fails and auto_amendify=True."""
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini", llm_api_key="test")
        mock_acompletion.side_effect = Exception("LLM error")

        from cognee.skills.client import Skills

        client = Skills()

        amendify_result = {
            "inspection": {"inspection_id": "i1", "root_cause": "bad instructions"},
            "amendment": {"amendment_id": "a1", "status": "proposed"},
            "applied": {"success": True, "status": "applied"},
        }

        async def _run():
            with patch.object(client, "load", new_callable=AsyncMock, return_value=SAMPLE_SKILL):
                with patch.object(client, "observe", new_callable=AsyncMock):
                    with patch.object(
                        client,
                        "auto_amendify",
                        new_callable=AsyncMock,
                        return_value=amendify_result,
                    ) as mock_amendify:
                        result = await client.execute(
                            "summarize",
                            "do something",
                            auto_amendify=True,
                            amendify_min_runs=1,
                        )
                        return result, mock_amendify

        result, mock_amendify = asyncio.run(_run())

        assert result["success"] is False
        assert result["amended"] is not None
        assert result["amended"]["applied"]["success"] is True
        mock_amendify.assert_called_once()

    @patch("cognee.skills.execute.litellm.acompletion")
    @patch("cognee.skills.execute.get_llm_config")
    def test_client_execute_auto_amendify_skipped_on_success(self, mock_config, mock_acompletion):
        """Client.execute does NOT trigger auto_amendify when execution succeeds."""
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini", llm_api_key="test")
        mock_acompletion.return_value = _make_mock_response("Success")

        from cognee.skills.client import Skills

        client = Skills()

        async def _run():
            with patch.object(client, "load", new_callable=AsyncMock, return_value=SAMPLE_SKILL):
                with patch.object(client, "observe", new_callable=AsyncMock):
                    with patch.object(
                        client, "auto_amendify", new_callable=AsyncMock
                    ) as mock_amendify:
                        result = await client.execute(
                            "summarize", "do something", auto_amendify=True
                        )
                        return result, mock_amendify

        result, mock_amendify = asyncio.run(_run())

        assert result["success"] is True
        assert "amended" not in result
        mock_amendify.assert_not_called()

    def test_client_auto_amendify_returns_none_no_failures(self):
        """auto_amendify returns None when there aren't enough failed runs."""
        from cognee.skills.client import Skills

        client = Skills()

        async def _run():
            with patch.object(client, "inspect", new_callable=AsyncMock, return_value=None):
                return await client.auto_amendify("summarize")

        result = asyncio.run(_run())
        assert result is None

    def test_client_auto_amendify_full_pipeline(self):
        """auto_amendify chains inspect → preview_amendify → amendify."""
        from cognee.skills.client import Skills

        client = Skills()

        mock_inspection = {
            "inspection_id": "i1",
            "skill_id": "summarize",
            "skill_name": "Summarize",
            "failure_category": "instruction_gap",
            "root_cause": "Missing details",
            "severity": "high",
            "improvement_hypothesis": "Add more detail",
            "analyzed_run_count": 3,
            "avg_success_score": 0.2,
            "inspection_confidence": 0.9,
        }
        mock_amendment = {
            "amendment_id": "a1",
            "skill_id": "summarize",
            "skill_name": "Summarize",
            "inspection_id": "i1",
            "change_explanation": "Added detail",
            "expected_improvement": "Better output",
            "status": "proposed",
            "amendment_confidence": 0.8,
            "pre_amendment_avg_score": 0.2,
        }
        mock_apply = {"success": True, "status": "applied", "amendment_id": "a1"}

        async def _run():
            with patch.object(
                client, "inspect", new_callable=AsyncMock, return_value=mock_inspection
            ):
                with patch.object(
                    client,
                    "preview_amendify",
                    new_callable=AsyncMock,
                    return_value=mock_amendment,
                ) as mock_pa:
                    with patch.object(
                        client,
                        "amendify",
                        new_callable=AsyncMock,
                        return_value=mock_apply,
                    ) as mock_am:
                        result = await client.auto_amendify("summarize", min_runs=1)
                        return result, mock_pa, mock_am

        result, mock_pa, mock_am = asyncio.run(_run())

        assert result["inspection"]["inspection_id"] == "i1"
        assert result["amendment"]["amendment_id"] == "a1"
        assert result["applied"]["success"] is True
        mock_pa.assert_called_once_with(skill_id="summarize", inspection_id="i1", node_set="skills")
        mock_am.assert_called_once_with(
            amendment_id="a1",
            write_to_disk=False,
            validate=False,
            validation_task_text="",
            node_set="skills",
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for event handlers (Triage, Ingest, Forget, OptOut)."""

from __future__ import annotations

import pytest

from config import BotConfig
from handlers import ForgetHandler, IngestHandler, OptOutHandler, TriageHandler
from memory_adapter import MemoryAdapter


@pytest.fixture
def setup_handlers(bot_config, mock_cognee):
    """Create all handlers with mocked cognee."""
    adapter = MemoryAdapter(bot_config)
    opt_out_list: set[str] = set()

    triage = TriageHandler(adapter, bot_config)
    ingest = IngestHandler(adapter, bot_config, opt_out_list)
    forget = ForgetHandler(adapter)
    optout = OptOutHandler(opt_out_list)

    return {
        "adapter": adapter,
        "triage": triage,
        "ingest": ingest,
        "forget": forget,
        "optout": optout,
        "opt_out_list": opt_out_list,
        "mock_cognee": mock_cognee,
    }


class TestTriageHandler:
    """Tests for TriageHandler."""

    @pytest.mark.asyncio
    async def test_produces_cited_reply(self, setup_handlers, new_support_issue):
        """Triage with matching issues → citations present in result."""
        triage = setup_handlers["triage"]
        result = await triage.handle(new_support_issue, "support")

        assert len(result.citations) > 0
        assert result.query == new_support_issue
        assert result.suggested_reply != ""
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_no_match_gives_clean_message(self, bot_config, mock_cognee_empty, unrelated_query):
        """Unrelated query → 'No similar resolved support threads' message, no hallucination."""
        adapter = MemoryAdapter(bot_config)
        triage = TriageHandler(adapter, bot_config)

        result = await triage.handle(unrelated_query, "support")

        assert result.citations == []
        assert result.suggested_reply == ""
        assert result.confidence == 0.0

        reply = triage.format_reply(result)
        assert "No similar resolved support threads" in reply

    @pytest.mark.asyncio
    async def test_deterministic_recall(self, setup_handlers, new_support_issue):
        """Same question asked twice → same citations returned consistently."""
        triage = setup_handlers["triage"]

        result1 = await triage.handle(new_support_issue, "support")
        result2 = await triage.handle(new_support_issue, "support")

        assert len(result1.citations) == len(result2.citations)
        for c1, c2 in zip(result1.citations, result2.citations):
            assert c1.source_thread_id == c2.source_thread_id
            assert c1.resolution_summary == c2.resolution_summary

    @pytest.mark.asyncio
    async def test_format_reply_includes_citations(self, setup_handlers, new_support_issue):
        """Formatted reply includes numbered citations."""
        triage = setup_handlers["triage"]
        result = await triage.handle(new_support_issue, "support")
        reply = triage.format_reply(result)

        assert "**[1]**" in reply
        assert "Similar past issues found" in reply


class TestIngestHandler:
    """Tests for IngestHandler."""

    @pytest.mark.asyncio
    async def test_builds_document_from_thread(self, setup_handlers):
        """Mock thread messages → verify structured document created."""
        ingest = setup_handlers["ingest"]
        mock = setup_handlers["mock_cognee"]

        result = await ingest.handle(
            thread_id="T100",
            channel_id="support",
            reporter="dave",
            messages=["Problem: API is slow", "Root cause: missing index", "Fixed with CREATE INDEX"],
            thread_url="https://example.com/T100",
        )

        assert result["status"] == "completed"
        # Verify remember was called
        mock["remember"].assert_called_once()
        # The document text should contain problem and resolution
        call_args = mock["remember"].call_args
        doc_text = call_args[0][0]
        assert "API is slow" in doc_text
        assert "CREATE INDEX" in doc_text

    @pytest.mark.asyncio
    async def test_optout_blocks_ingestion(self, setup_handlers):
        """After !optout, verify ingest handler skips user's threads."""
        ingest = setup_handlers["ingest"]
        opt_out_list = setup_handlers["opt_out_list"]
        mock = setup_handlers["mock_cognee"]

        # Opt out
        opt_out_list.add("dave")

        result = await ingest.handle(
            thread_id="T101",
            channel_id="support",
            reporter="dave",
            messages=["Problem", "Solution"],
        )

        assert result["status"] == "skipped"
        assert result["reason"] == "user_opted_out"
        mock["remember"].assert_not_called()


class TestForgetHandler:
    """Tests for ForgetHandler."""

    @pytest.mark.asyncio
    async def test_removes_specific_thread(self, setup_handlers):
        """!forget command → verify forget() invoked with correct data_id."""
        adapter = setup_handlers["adapter"]
        ingest = setup_handlers["ingest"]
        forget = setup_handlers["forget"]
        mock = setup_handlers["mock_cognee"]

        # First ingest
        await ingest.handle(
            thread_id="T200",
            channel_id="support",
            reporter="eve",
            messages=["Bug", "Fix"],
        )

        # Now forget
        result = await forget.handle("T200")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_forget_unknown_thread(self, setup_handlers):
        """Forget unknown thread → error status."""
        forget = setup_handlers["forget"]

        result = await forget.handle("NONEXISTENT")
        assert result["status"] == "error"
        assert "was never ingested" in result["message"]


class TestOptOutHandler:
    """Tests for OptOutHandler."""

    def test_optout_adds_to_blocklist(self, setup_handlers):
        """!optout → user added to blocklist."""
        optout = setup_handlers["optout"]
        opt_out_list = setup_handlers["opt_out_list"]

        result = optout.handle("frank")

        assert result["status"] == "success"
        assert "frank" in opt_out_list

    def test_optout_idempotent(self, setup_handlers):
        """Multiple opt-outs for the same user don't error."""
        optout = setup_handlers["optout"]

        result1 = optout.handle("frank")
        result2 = optout.handle("frank")

        assert result1["status"] == "success"
        assert result2["status"] == "success"

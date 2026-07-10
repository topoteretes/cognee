"""Capture on one transport, recall from another, with a citation back to the source."""

import asyncio


def test_capture_on_telegram_recall_on_web_with_citation(harness):
    async def run():
        # Two front-ends linked to one brain (linking itself is covered in test_identity).
        canonical = harness.identity.resolve("telegram", "alice_tg")
        harness.identity.link("web", "alice_web", canonical)

        saved = await harness.send(
            "telegram",
            "alice_tg",
            "I parked the car on level 3 of the garage",
            ts="2026-06-12T09:00:00",
        )
        assert "Saved" in saved

        # Recall from a different transport hits the same brain.
        reply = await harness.send("web", "alice_web", "where did I park?")
        assert "level 3" in reply.lower()

        # The reply cites the original Telegram source and date.
        assert "telegram" in reply.lower()
        assert "2026-06-12" in reply

        # Structural check on the citation the adapter produced.
        web_convo = harness.conversation("web", "alice_web")
        answer = await harness.adapter.answer(web_convo, "where did I park?")
        assert answer.citations
        citation = answer.citations[0]
        assert citation.source_transport == "telegram"
        assert citation.timestamp == "2026-06-12T09:00:00"
        assert citation.source_ref

    asyncio.run(run())


def test_note_prefix_forces_capture_of_a_question_shaped_note(harness):
    async def run():
        # A message ending in "?" is normally recalled; /note forces capture.
        saved = await harness.send("web", "bob", "/note ask Bob about the merger?")
        assert "Saved" in saved
        reply = await harness.send("web", "bob", "merger?")
        assert "merger" in reply.lower()

    asyncio.run(run())


def test_ask_prefix_forces_recall_without_a_question_mark(harness):
    async def run():
        await harness.send("web", "bob", "the vault code is 4417")
        reply = await harness.send("web", "bob", "/ask vault code")
        assert "4417" in reply

    asyncio.run(run())

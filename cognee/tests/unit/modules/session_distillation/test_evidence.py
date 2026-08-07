from datetime import datetime, timezone

import pytest

from cognee.infrastructure.session.session_search_models import SessionTurnEvidence
from cognee.modules.session_distillation.evidence import (
    build_evidence_batches,
    load_eligible_evidence,
    mark_evidence_distilled,
)
from cognee.modules.session_distillation.models import (
    CURATOR_BLOCKS_PER_BATCH,
    MAX_CANDIDATE_CHARS,
)


def _evidence(evidence_id: str, **overrides) -> dict:
    values = {
        "id": evidence_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "dataset-1",
        "current_raw_message": "Remember this.",
        "current_response": "Okay.",
        "feedback_evidence": ["This helped."],
        "future_context_evidence": ["Prefer concise answers."],
    }
    values.update(overrides)
    return SessionTurnEvidence(**values).model_dump(mode="json")


class FakeSessionManager:
    def __init__(self, rows, failed_updates=None):
        self.rows = list(rows)
        self.failed_updates = set(failed_updates or [])

    async def get_session_context_entries(self, user_id, session_id, strict=False):
        return list(self.rows)

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        if entry_id in self.failed_updates:
            return False
        for row in self.rows:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False


def test_load_selects_only_recoverable_dataset_evidence():
    rows = [
        _evidence("pending"),
        _evidence("deferred", status="deferred"),
        _evidence("failed", status="failed"),
        _evidence("completed", status="completed"),
        _evidence("distilled", distilled_at="2026-01-01T00:00:00+00:00"),
        _evidence("other-dataset", dataset_id="dataset-2"),
        _evidence("unscoped", dataset_id=None),
        _evidence("no-claims", feedback_evidence=[], future_context_evidence=[]),
        {"id": "context", "kind": "context"},
        "not-a-row",
    ]

    eligible = load_eligible_evidence(
        rows,
        dataset_id="dataset-1",
        tracked_evidence_ids={"deferred"},
    )

    assert [entry.id for entry in eligible] == ["pending", "failed"]


def test_evidence_batch_normalizes_and_deduplicates_each_evidence_kind():
    entry = SessionTurnEvidence.model_validate(
        _evidence(
            "e1",
            feedback_evidence=["  Same   claim  ", "same claim", "Other"],
            future_context_evidence=[" Future   claim ", "future claim", "x" * 500],
        )
    )

    batch = build_evidence_batches([entry])[0]

    assert batch.source_evidence_ids == ("e1",)
    assert batch.text.count("Same claim") == 1
    assert batch.text.count("Future claim") == 1
    assert "Feedback evidence:" in batch.text
    assert "Future-context evidence:" in batch.text
    assert "x" * MAX_CANDIDATE_CHARS in batch.text


def test_evidence_batches_are_chronological_and_size_bounded():
    evidence = [
        SessionTurnEvidence.model_validate(
            _evidence(
                f"e{index}",
                created_at=f"2026-08-06T10:00:{index:02d}+00:00",
                feedback_evidence=[f"Claim {index}."],
            )
        )
        for index in reversed(range(CURATOR_BLOCKS_PER_BATCH + 1))
    ]

    batches = build_evidence_batches(evidence)

    assert len(batches) == 2
    assert batches[0].source_evidence_ids == tuple(
        f"e{index}" for index in range(CURATOR_BLOCKS_PER_BATCH)
    )
    assert batches[1].source_evidence_ids == (f"e{CURATOR_BLOCKS_PER_BATCH}",)
    assert batches[0].text.index("Claim 0.") < batches[0].text.index("Claim 1.")


def test_empty_evidence_produces_no_batches():
    assert build_evidence_batches([]) == []


@pytest.mark.asyncio
async def test_mark_rechecks_and_skips_consumed_or_unwritable_evidence():
    manager = FakeSessionManager(
        [
            _evidence("e1"),
            _evidence("e2", status="completed"),
            _evidence("e3"),
        ],
        failed_updates={"e3"},
    )

    marked = await mark_evidence_distilled(
        manager,
        user_id="user",
        session_id="session",
        dataset_id="dataset-1",
        evidence_ids={"e1", "e2", "e3", "missing"},
    )

    assert marked == {"e1"}
    assert manager.rows[0]["distilled_at"] is not None
    assert manager.rows[1]["distilled_at"] is None
    assert manager.rows[2]["distilled_at"] is None


@pytest.mark.asyncio
async def test_mark_fails_open_when_the_cache_read_raises():
    manager = FakeSessionManager([_evidence("e1")])

    async def fail_read(**kwargs):
        raise RuntimeError("cache unavailable")

    manager.get_session_context_entries = fail_read

    assert (
        await mark_evidence_distilled(
            manager,
            user_id="user",
            session_id="session",
            dataset_id="dataset-1",
            evidence_ids={"e1"},
        )
        == set()
    )
    assert manager.rows[0]["distilled_at"] is None

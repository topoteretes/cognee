"""review_conflicts on a graph that records its writes in the relational ledger, on SQLite.

A graph that held data before its first provenance-aware write (one created before
cognee 1.5), or a backend without ``set_graph_metadata`` (Turso, Neptune), keeps its
rollback provenance in the relational ``nodes`` / ``edges`` tables. The conflict review's
writes go through ``add_data_points`` into those tables too. Offline: the LLM and the
embeddings are the journeys' deterministic mocks.
"""

import os
import re
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import func, select

import cognee
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.engine.models import Entity
from cognee.modules.engine.operations.setup import setup as create_databases
from cognee.modules.graph.models import Edge
from cognee.modules.graph.utils.fact_conflicts import (
    CONFLICT_ABOUT,
    CONFLICT_CITES,
    CONFLICT_VALUE,
)
from cognee.tasks.memify.review_conflicts.schema import ReportedConflict, ReviewResponse
from cognee.tests.journeys import mock_ai


def version_graph(version: str) -> dict:
    """One document's graph: the timer and the RotorHazard version it runs."""
    value = f"RotorHazard {version}"
    return {
        "nodes": [
            {"id": "rh1", "name": "rh1", "type": "RaceTimer", "description": "Race timer"},
            {"id": value, "name": value, "type": "SoftwareVersion", "description": value},
        ],
        "edges": [
            {"source_node_id": "rh1", "target_node_id": value, "relationship_name": "runs_version"}
        ],
    }


DATASET = "conflict_trial"
VERSIONS = {"rh1 timer": "4.4.0", "rh1 timer upgrade": "4.5.0"}
DOCUMENTS = {
    title: f"Title: {title}\n\nrh1 (Raspberry Pi 5 race timer) runs RotorHazard {version}."
    for title, version in VERSIONS.items()
}
MOCK_GRAPHS = {
    title: {"knowledge_graph": version_graph(version)} for title, version in VERSIONS.items()
}
# Review prompt lines: "Subject [n1] rh1 (types: ...)" and "[r1] n1 --runs_version--> n2 | ...".
SUBJECT_LABEL = re.compile(r"^Subject \[(n\d+)\]", re.MULTILINE)
CLAIM_LINE = re.compile(r"\[(r\d+)\] (n\d+) --\S+--> (n\d+)")


@pytest_asyncio.fixture
async def ledger_env(tmp_path):
    """Fresh roots under tmp_path, one graph shared by every dataset, mocked AI calls."""
    with patch.dict(os.environ, {"ENABLE_BACKEND_ACCESS_CONTROL": "false"}):
        mock_ai.install_all(MOCK_GRAPHS)
        install_mock_review()
        clear_config_caches()
        previous_roots = await use_fresh_roots(tmp_path)
        try:
            yield
        finally:
            await restore_roots(previous_roots)
            mock_ai.uninstall_all()
    clear_config_caches()


async def use_fresh_roots(tmp_path) -> tuple[str, str]:
    """Point cognee at empty data and system roots; return the previous ones."""
    base_config = get_base_config()
    previous_roots = (base_config.data_root_directory, base_config.system_root_directory)
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    clear_engine_caches()
    await prune_everything()
    await create_databases()
    return previous_roots


async def restore_roots(previous_roots: tuple[str, str]) -> None:
    """Wipe the temporary stores and point cognee back at the previous roots."""
    await prune_everything()
    cognee.config.data_root_directory(previous_roots[0])
    cognee.config.system_root_directory(previous_roots[1])
    clear_engine_caches()


async def prune_everything() -> None:
    """Delete the data and the system databases through the engines open on the roots."""
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def clear_engine_caches() -> None:
    """Forget engines opened on earlier roots, as the journeys do between tests."""
    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()


def clear_config_caches() -> None:
    """Re-read LLM and embedding settings: both are cached at import, before the mock keys."""
    get_llm_config.cache_clear()
    get_embedding_config.cache_clear()


async def seed_graph_before_first_cognify() -> None:
    """Give the graph data before cognee first writes to it, as a pre-1.5 store has.

    A graph that is not empty is never marked graph-provenance, so every later write
    keeps recording its provenance in the relational ledger.
    """
    graph_engine = await get_graph_engine()
    await graph_engine.add_nodes([Entity(name="Pre-1.5 entity", description="Seed")])


async def cognify_conflicting_documents() -> None:
    """Add the two documents that give rh1 different versions, then build the graph."""
    await cognee.add(list(DOCUMENTS.values()), dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])


def find_stage(result, name: str):
    """The named stage's result; on a failed assertion pytest shows its reason and error."""
    return next(stage for stage in result.stages if stage.stage == name)


async def count_ledger_edges(relationship_names: list[str]) -> dict[str, int]:
    """Rows per relationship name in the relational ledger's ``edges`` table."""
    async with get_relational_engine().get_async_session() as session:
        rows = await session.execute(
            select(Edge.relationship_name, func.count())
            .where(Edge.relationship_name.in_(relationship_names))
            .group_by(Edge.relationship_name)
        )
        return dict(rows.all())


def install_mock_review() -> None:
    """Answer conflict review calls; every other call goes to the installed mock."""
    answer_other_calls = LLMGateway.acreate_structured_output

    async def acreate(text_input, system_prompt, response_model, **kwargs):
        if response_model is ReviewResponse:
            return conflicting_review(text_input)
        return await answer_other_calls(text_input, system_prompt, response_model, **kwargs)

    LLMGateway.acreate_structured_output = staticmethod(acreate)


def conflicting_review(prompt: str) -> ReviewResponse:
    """Report each subject's claims as one fixed conflict, read from the prompt's labels."""
    subjects = SUBJECT_LABEL.findall(prompt)
    claims = CLAIM_LINE.findall(prompt)
    return ReviewResponse(
        reviewed=subjects,
        conflicts=[
            ReportedConflict(
                about=subject,
                attribute="runs_version",
                kind="fixed",
                text="rh1 is reported with two RotorHazard versions.",
                conflicting=[label for label, *ends in claims if subject in ends],
            )
            for subject in subjects
        ],
    )


@pytest.mark.asyncio
async def test_review_conflicts_records_conflict_edges_in_the_ledger(ledger_env):
    await seed_graph_before_first_cognify()
    await cognify_conflicting_documents()

    result = await cognee.improve(dataset=DATASET, review_conflicts=True)

    assert find_stage(result, "review_conflicts").status == "completed"
    assert await count_ledger_edges([CONFLICT_ABOUT, CONFLICT_VALUE, CONFLICT_CITES]) == {
        CONFLICT_ABOUT: 1,
        CONFLICT_VALUE: 2,
        CONFLICT_CITES: 2,
    }

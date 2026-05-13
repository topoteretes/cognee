"""
Smoke e2e test for feedback-weight memify pipeline.
"""

import asyncio
import os
import pathlib

import cognee
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import create_user, get_default_user, get_user_by_email
from cognee.tasks.memify.feedback_weights_constants import (
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)


async def main():
    base_dir = pathlib.Path(__file__).parent
    data_directory_path = str(
        (base_dir / ".data_storage/test_feedback_weights_memify_pipeline").resolve()
    )
    cognee_directory_path = str(
        (base_dir / ".cognee_system/test_feedback_weights_memify_pipeline").resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "feedback_weights_pipeline_smoke"
    await cognee.add(
        data=(
            "Cognee builds knowledge graphs from text. TechCorp focuses on AI and machine learning."
        ),
        dataset_name=dataset_name,
    )
    await cognee.cognify(datasets=[dataset_name])

    user = await get_default_user()
    session_id = "feedback_pipeline_smoke_session"

    await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What does TechCorp focus on?",
        session_id=session_id,
        top_k=1,
    )

    session_manager = get_session_manager()
    entries = await session_manager.get_session(
        user_id=str(user.id),
        session_id=session_id,
        formatted=False,
    )
    assert entries, "Expected at least one QA entry in session."

    qa_entry = None
    for entry in entries:
        ids = entry.used_graph_element_ids
        if isinstance(ids, dict) and (ids.get("node_ids") or ids.get("edge_ids")):
            qa_entry = entry
            break

    assert qa_entry is not None, "Expected a QA entry with used_graph_element_ids."

    ok = await cognee.session.add_feedback(
        session_id=session_id,
        qa_id=qa_entry.qa_id,
        feedback_text="Helpful answer",
        feedback_score=5,
        user=user,
    )
    assert ok is True, "Failed to attach feedback to QA entry."

    await apply_feedback_weights_pipeline(
        user=user,
        session_ids=[session_id],
        dataset=dataset_name,
        alpha=0.1,
        batch_size=10,
        run_in_background=False,
    )

    updated_entries = await session_manager.get_session(
        user_id=str(user.id),
        session_id=session_id,
        formatted=False,
    )
    updated_qa = next((entry for entry in updated_entries if entry.qa_id == qa_entry.qa_id), None)
    assert updated_qa is not None, "Updated QA entry not found."
    assert updated_qa.memify_metadata.get(MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY) is True, (
        f"Pipeline should mark {MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY}=True on successful QA update."
    )

    graph_engine = await get_graph_engine()
    used_ids = updated_qa.used_graph_element_ids
    node_ids = [
        node_id for node_id in used_ids.get("node_ids", []) if isinstance(node_id, str) and node_id
    ]
    edge_ids = [
        edge_id for edge_id in used_ids.get("edge_ids", []) if isinstance(edge_id, str) and edge_id
    ]

    node_weights = await graph_engine.get_node_feedback_weights(node_ids) if node_ids else {}
    edge_weights = await graph_engine.get_edge_feedback_weights(edge_ids) if edge_ids else {}

    assert any(abs(float(weight) - 0.5) > 1e-9 for weight in node_weights.values()) or any(
        abs(float(weight) - 0.5) > 1e-9 for weight in edge_weights.values()
    ), "Expected at least one non-default feedback_weight after pipeline run."

    secondary_email = "feedback.pipeline.noaccess@example.com"
    secondary_user = await get_user_by_email(secondary_email)
    if secondary_user is None:
        secondary_user = await create_user(
            email=secondary_email,
            password="feedback_pipeline_noaccess_password",
            is_superuser=False,
            is_active=True,
            is_verified=True,
        )

    try:
        await apply_feedback_weights_pipeline(
            user=secondary_user,
            session_ids=[session_id],
            dataset=dataset_name,
            alpha=0.1,
            batch_size=10,
            run_in_background=False,
        )
        raise AssertionError(
            "Expected CogneeValidationError for user without dataset write access."
        )
    except CogneeValidationError as error:
        assert "does not have write access to dataset" in str(error)


if __name__ == "__main__":
    os.environ.setdefault("ENV", "dev")
    asyncio.run(main())

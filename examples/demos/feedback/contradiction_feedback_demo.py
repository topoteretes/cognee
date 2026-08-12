"""Contradiction detection + feedback, live and visualized step by step.

Runs the real pipeline on two conflicting one-line documents:

  1. remember  "Anna leads Falcon. Budget is 2M EUR."
  2. remember  "Marko leads Falcon. Budget is 5M EUR."  -> contradicts edge
  3. ask       "What is the budget of Project Falcon?"  -> both facts retrieved
  4. feedback  5/5 + comment                             -> feedback weights shift
  5. ask again -> still reports the conflict: a rating can't pick a winner
  6. remember the correction -> the answer flips: new knowledge decides truth

Each step prints the ACTUAL state read back from the graph / session store.
Storage is isolated under /tmp/conflict_demo so it never touches real data.
Requires a working LLM + embedding config (.env) — run from the repo root:

    uv run python examples/demos/feedback/contradiction_feedback_demo.py
"""

import asyncio
import os
import shutil
from pathlib import Path

DEMO_ROOT = Path("/tmp/conflict_demo")
shutil.rmtree(DEMO_ROOT, ignore_errors=True)

# Env must be set before cognee is imported: isolated storage, detection on,
# session cache on (records which graph elements each answer used), and a
# non-zero feedback influence so ratings actually affect future ranking.
os.environ.update(
    {
        "DATA_ROOT_DIRECTORY": str(DEMO_ROOT / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(DEMO_ROOT / "system"),
        "CONTRADICTION_DETECTION": "true",
        "CACHING": "true",
        "DEFAULT_FEEDBACK_INFLUENCE": "0.2",
    }
)

import cognee  # noqa: E402
from cognee import SearchType  # noqa: E402
from cognee.infrastructure.databases.graph import get_graph_engine  # noqa: E402
from cognee.infrastructure.session.get_session_manager import get_session_manager  # noqa: E402
from cognee.memify_pipelines.apply_feedback_weights import (  # noqa: E402
    apply_feedback_weights_pipeline,
)
from cognee.modules.users.methods import get_default_user  # noqa: E402

WIDTH = 78
DATASET = "falcon_demo"
SESSION = "board_demo_session"
QUESTION = "What is the budget of Project Falcon?"

# Edges that describe graph structure rather than semantic facts; hidden so the
# visualization shows only the human-meaningful statements.
STRUCTURAL = {"contains", "is_part_of", "made_from", "exists_in"}


def first_answer(results) -> str:
    """Pull the completion text out of a recall result list."""
    if not results:
        return "(no results)"
    item = results[0]
    if hasattr(item, "text"):
        return str(item.text)
    if isinstance(item, dict) and item.get("search_result"):
        return str(item["search_result"][0])
    return str(item)


def show(title: str, lines: list) -> None:
    """Print one step as a fixed-width ASCII box."""
    inner = WIDTH - 2
    pad = inner - len(title) - 2
    print("+" + "-" * (pad // 2) + f" {title} " + "-" * (pad - pad // 2) + "+")
    for line in lines:
        # LLM answers may contain newlines; each rendered row must stay boxed.
        for row in str(line).splitlines() or [""]:
            for chunk in [row[i : i + inner - 2] for i in range(0, max(len(row), 1), inner - 2)]:
                print("| " + chunk.ljust(inner - 2) + " |")
    print("+" + "-" * inner + "+")
    print()


async def read_facts():
    """Read every semantic fact and every contradicts edge back from the graph."""
    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    names = {str(node_id): props.get("name", str(node_id)[:8]) for node_id, props in nodes}

    facts, conflicts = [], []
    for source, target, relationship, props in edges:
        if relationship == "contradicts":
            conflicts.append(props)
        elif relationship not in STRUCTURAL:
            facts.append(
                f"({names.get(str(source))}) --{relationship}--> ({names.get(str(target))})"
            )
    return facts, conflicts


async def element_weights(used_ids: "dict | None") -> dict:
    """Current feedback weights of the graph elements one answer used."""
    graph = await get_graph_engine()
    weights = {}
    node_ids = (used_ids or {}).get("node_ids") or []
    edge_ids = (used_ids or {}).get("edge_ids") or []
    if node_ids:
        found = await graph.get_node_feedback_weights(node_ids)
        weights.update({f"node {k[:13]}": v for k, v in found.items()})
    if edge_ids:
        found = await graph.get_edge_feedback_weights(edge_ids)
        weights.update({f"edge {k[:13]}": v for k, v in found.items()})
    return weights


async def main() -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # ---- STEP 1: first document ------------------------------------------ #
    await cognee.remember(
        "Anna leads Project Falcon. The budget of Project Falcon is 2 million euros.",
        dataset_name=DATASET,
        self_improvement=False,
    )
    facts, conflicts = await read_facts()
    show(
        "STEP 1  REMEMBER: 'Anna leads Falcon. Budget is 2M EUR.'",
        ["facts now in the knowledge graph:", ""]
        + [f"   {f}" for f in facts]
        + ["", f"contradictions flagged: {len(conflicts)}"],
    )

    # ---- STEP 2: conflicting document ------------------------------------ #
    await cognee.remember(
        "Marko leads Project Falcon. The budget of Project Falcon is 5 million euros.",
        dataset_name=DATASET,
        self_improvement=False,
    )
    facts, conflicts = await read_facts()
    conflict_lines = []
    for conflict in conflicts:
        conflict_lines += [
            "",
            f"   FACT A : {conflict.get('first_fact')}",
            f"   FACT B : {conflict.get('second_fact')}",
            f"   reason : {conflict.get('reason')}",
            f"   confidence: {conflict.get('confidence')}",
        ]
    show(
        "STEP 2  REMEMBER: 'Marko leads Falcon. Budget is 5M EUR.'",
        ["facts now in the knowledge graph:", ""]
        + [f"   {f}" for f in facts]
        + ["", f"contradictions flagged: {len(conflicts)} (nothing was deleted)"]
        + conflict_lines,
    )

    # ---- STEP 3: ask — retrieval sees both sides -------------------------- #
    results = await cognee.recall(
        QUESTION,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        session_id=SESSION,
    )
    answer = first_answer(results)

    user = await get_default_user()
    qa_entries = await get_session_manager().get_session(user_id=str(user.id), session_id=SESSION)
    assert isinstance(qa_entries, list) and qa_entries, "session recorded no QA entry"
    qa = qa_entries[-1]
    weights_before = await element_weights(qa.used_graph_element_ids)
    show(
        "STEP 3  ASK: 'What is the budget of Project Falcon?'",
        ["answer:", f"   {answer}", ""]
        + [f"graph elements used by this answer: {len(weights_before)}"]
        + [f"   {element}: weight {weight}" for element, weight in list(weights_before.items())[:6]]
        + ["", f"recorded in session '{SESSION}' as qa_id {str(qa.qa_id)[:8]}..."],
    )

    # ---- STEP 4: feedback closes the loop --------------------------------- #
    await cognee.session.add_feedback(
        session_id=SESSION,
        qa_id=qa.qa_id,
        feedback_score=5,
        feedback_text="Correct — 5 million is the approved budget; Marko took over in June.",
    )
    await apply_feedback_weights_pipeline(
        user=user, session_ids=[SESSION], dataset=DATASET, alpha=0.1
    )
    weights_after = await element_weights(qa.used_graph_element_ids)
    show(
        "STEP 4  FEEDBACK: rated 5/5 -> weights shift",
        ["feedback weight per element (before -> after):", ""]
        + [
            f"   {element}: {weights_before.get(element)} -> {weights_after.get(element)}"
            for element in list(weights_after)[:6]
        ]
        + [
            "",
            "high-rated elements now rank higher in future searches",
            "(DEFAULT_FEEDBACK_INFLUENCE=0.2 blends weight into retrieval scoring);",
            "both original facts and the contradicts edge remain stored.",
        ],
    )

    # ---- STEP 5: ask again — a rating alone can't pick a winner ----------- #
    # The 5/5 rating up-weighted every element the answer used, INCLUDING both
    # budget facts (the answer needed both to report the conflict). A symmetric
    # signal cannot break the tie, so the answer still reports the conflict.
    results = await cognee.recall(
        QUESTION,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        session_id="fresh_session_1",
    )
    show(
        "STEP 5  ASK AGAIN: a rating alone cannot pick a winner",
        ["answer (fresh session, graph + weights only):", f"   {first_answer(results)}", ""]
        + [
            "both budget facts were up-weighted equally (both were used by the",
            "rated answer), so ranking between them is unchanged -- ratings",
            "steer which memories get attention; they never decide what is true.",
        ],
    )

    # ---- STEP 6: the correction becomes memory -> answer flips ------------ #
    await cognee.remember(
        "The approved budget of Project Falcon is 5 million euros. "
        "The earlier 2 million euro figure is outdated.",
        dataset_name=DATASET,
        self_improvement=False,
    )
    _, conflicts = await read_facts()
    results = await cognee.recall(
        QUESTION,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        session_id="fresh_session_2",
    )
    show(
        "STEP 6  REMEMBER THE CORRECTION -> the answer flips",
        [
            "new document: 'The approved budget is 5M EUR; the 2M figure is",
            "outdated.' (this is what textual feedback becomes when persisted)",
            "",
            "answer (fresh session):",
            f"   {first_answer(results)}",
            "",
            f"contradictions now flagged in the graph: {len(conflicts)}",
            "the 2M fact is still stored (auditable), but retrieval now has a",
            "correction that explicitly supersedes it -- new knowledge, not the",
            "rating, is what changed the answer.",
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())

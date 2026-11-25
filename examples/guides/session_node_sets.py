"""Keep each project's session memory apart with a node_set pinned on the session.

Two sessions, two projects. Each session's first write carries a node_set, which
pins it on the session; improve() copies the session into the graph and keeps
the set on everything it bridges (Q&A, agent traces, distilled lessons). A recall
scoped with node_name then sees only that project's memory.

Rules shown here:
- The node_set field on a QAEntry or TraceEntry and the call-level
  remember(..., node_set=[...]) kwarg are the same thing; plain text pins too.
- Later writes may repeat the set or omit it. A different set raises
  cognee.SessionNodeSetConflictError (HTTP 409 over the API), so start a new
  session for a different project.
- The list must hold non-empty names; it is stored sorted and deduplicated. An
  empty list pins nothing.

Deployment note: the pin is checked under an in-process lock, so the one-set-
per-session guarantee holds within one worker process. With several workers,
route a session's writes to one worker; two workers taking the same session's
first write at the same moment can each pin a set, and the last write wins.

Requires: LLM_API_KEY.
Run: uv run python examples/guides/session_node_sets.py
"""

import asyncio

import cognee
from cognee import SearchType

DATASET = "session_node_sets_guide"
QUESTION = "Who maintains the release checklist?"


async def main():
    # Start clean (optional in your app)
    await cognee.forget(everything=True)

    # A dataset for the sessions to bridge into.
    await cognee.remember("The office closes at six.", dataset_name=DATASET, self_improvement=False)

    # Project A: the node_set on the first typed entry pins the session.
    await cognee.remember(
        cognee.QAEntry(
            question=QUESTION,
            answer="In project A the release checklist is maintained by Ingrid.",
            node_set=["project-a"],
        ),
        dataset_name=DATASET,
        session_id="session-project-a",
    )
    # An agent step in the same session: repeating the set is fine.
    await cognee.remember(
        cognee.TraceEntry(
            origin_function="find_checklist_owner",
            status="success",
            method_return_value={"owner": "Ingrid"},
            node_set=["project-a"],
        ),
        dataset_name=DATASET,
        session_id="session-project-a",
    )

    # Project B: plain text with the call-level kwarg pins the second session.
    stored = await cognee.remember(
        "In project B the release checklist is maintained by Teodor.",
        dataset_name=DATASET,
        session_id="session-project-b",
        node_set=["project-b"],
    )
    await stored  # let the background session bridge finish before improving

    # A session never spans two projects.
    try:
        await cognee.remember(
            cognee.QAEntry(question="q", answer="a", node_set=["project-b"]),
            dataset_name=DATASET,
            session_id="session-project-a",
        )
    except cognee.SessionNodeSetConflictError as error:
        print("Refused:", error)

    # Bridge both sessions into the graph; each keeps its pinned set.
    await cognee.improve(DATASET, session_ids=["session-project-a", "session-project-b"])

    for project in ("project-a", "project-b"):
        results = await cognee.recall(
            QUESTION,
            datasets=[DATASET],
            query_type=SearchType.GRAPH_COMPLETION,
            node_name=[project],
        )
        print(f"{project}:", results[0].text if results else "(nothing found)")

    await cognee.wait_for_background_tasks()


if __name__ == "__main__":
    asyncio.run(main())

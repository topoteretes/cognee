"""Text, code, and session distillation with Cognee 1.5.4.

    python -m pip install "cognee==1.5.4"
    export LLM_API_KEY="YOUR_OPENAI_API_KEY"
    python examples/demos/company_brain_demo.py
    python examples/demos/company_brain_demo.py --recall-only

Requires an LLM and embedding provider for text, session learning, and answers.
Code extraction uses Enola (downloaded automatically on first use). The tiny code
fixture and default storage live beside this script under .cognee-readme-demo.
Existing storage environment variables take precedence. No memory is deleted.
Distillation is model-dependent; the script reports when no lesson is published.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

DATASET = "company_brain_readme_demo"
DOCUMENT = "Alice maintains the payments API. The payments API uses PostgreSQL."
LESSON = (
    "For every payments API release, always run the replay test before deploying. "
    "We learned this after duplicate webhook delivery caused a double charge. "
    "Remember this as a standing release rule for the payments API."
)
QUESTION = (
    "Who maintains the payments API, which database does it use, "
    "and what release rule did we learn?"
)
CODE = '''"""Payments API: guard against duplicate webhook delivery."""


def charge_once(event_id: str, processed_events: set[str]) -> bool:
    """Return False when this event has already been processed."""
    if event_id in processed_events:
        return False
    processed_events.add(event_id)
    return True


def replay_test() -> None:
    """Replaying the same event must not produce a second charge."""
    events: set[str] = set()
    assert charge_once("webhook-42", events)
    assert not charge_once("webhook-42", events)
'''


def prepare_demo():
    # Load provider settings once, before choosing defaults for demo storage.
    from dotenv import load_dotenv

    load_dotenv(override=False)
    # python-dotenv 1.2+ honors this switch, preserving the configuration below
    # when Cognee imports dotenv again.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    root = Path(__file__).resolve().parent / ".cognee-readme-demo"
    for variable, directory in (
        ("DATA_ROOT_DIRECTORY", "data"),
        ("SYSTEM_ROOT_DIRECTORY", "system"),
        ("CACHE_ROOT_DIRECTORY", "cache"),
        ("COGNEE_LOGS_DIR", "logs"),
    ):
        os.environ.setdefault(variable, str(root / directory))
    # Feedback analysis captures durable guidance during the scripted session.
    os.environ["CACHING"] = "true"
    os.environ["AUTO_FEEDBACK"] = "true"
    return root


def print_answers(entries):
    for entry in entries:
        print(entry.text)


async def recall_saved_memory(cognee):
    print("\nAnswer in a fresh session:", flush=True)
    print_answers(
        await cognee.recall(
            QUESTION,
            query_type=cognee.SearchType.RAG_COMPLETION,
            datasets=[DATASET],
            session_id=f"readme-verification-{uuid4().hex}",
        )
    )


async def index_code(cognee, root):
    repo = root / "payments-example"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "payments.py").write_text(CODE)
    print("\nIndexing the sample code:", flush=True)
    await cognee.remember(str(repo), dataset_name=DATASET, content_type="code")
    facts = await cognee.search(
        query_type=cognee.SearchType.CODE,
        query_text="",
        datasets=[DATASET],
        code_query={"operation": "query_facts", "kinds": ["symbol"], "limit": 10},
    )
    print(json.dumps(facts, indent=2, default=str))


async def main(args):
    root = prepare_demo()
    import cognee

    if args.recall_only:
        await recall_saved_memory(cognee)
        return
    if args.code_only:
        await index_code(cognee, root)
        return

    print("1. Remember the document.", flush=True)
    await cognee.remember(DOCUMENT, dataset_name=DATASET, self_improvement=False)

    print("2. Build and query a code graph.", flush=True)
    await index_code(cognee, root)

    print("3. Learn a release rule during a session.", flush=True)
    session_id = f"readme-learning-{uuid4().hex}"
    await cognee.recall(
        "Who maintains the payments API?",
        query_type=cognee.SearchType.RAG_COMPLETION,
        datasets=[DATASET],
        session_id=session_id,
    )
    await cognee.recall(
        LESSON,
        query_type=cognee.SearchType.RAG_COMPLETION,
        datasets=[DATASET],
        session_id=session_id,
    )

    print("4. Distill the session into permanent memory.", flush=True)
    distilled = await cognee.session.distill_session(session_id, dataset=DATASET)
    print(f"Distillation: {distilled.status}; {len(distilled.documents)} lesson documents")
    for document in distilled.documents:
        print(document)
    if distilled.status != "completed" or not distilled.documents:
        raise RuntimeError(
            "No new lesson was published. Inspect the distillation status and provider logs. "
            "An existing equivalent lesson can also cause the curator to reject a duplicate."
        )

    print("5. Recall the document and learned rule in a fresh session.", flush=True)
    await recall_saved_memory(cognee)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--recall-only", action="store_true", help="Query previously stored memory.")
    mode.add_argument(
        "--code-only", action="store_true", help="Try code extraction without LLM calls."
    )
    asyncio.run(main(parser.parse_args()))

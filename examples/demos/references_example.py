"""Example: lightweight references (Evidence) in search answers.

Demonstrates the `include_references` flag added to `cognee.search(...)`:

- RAG_COMPLETION  -> chunk evidence built from retrieved vector payloads
- GRAPH_COMPLETION -> entity/chunk fallback evidence walked from the graph
- include_references=False -> the original concise answer, no Evidence section

Runs fully self-contained on the default local stack (Ladybug graph, LanceDB
vector, SQLite relational) in an isolated data directory, so it does not touch
your configured databases. Requires only LLM_API_KEY (OpenAI) in the env.
"""

import os
import asyncio
import tempfile
from pathlib import Path

_DATA_DIR = tempfile.mkdtemp(prefix="cognee_references_example_")
# Set these before Cognee config is initialized so the example uses the intended values.
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["CACHING"] = "false"

import cognee  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402

# Force the default local stack via the config API. cognee loads .env with
# override=True (which here points the graph at a remote turbopuffer instance),
# so env vars alone are not enough — the config setters win.
cognee.config.set_graph_database_provider("kuzu")  # Ladybug (local, embedded)
cognee.config.set_vector_db_provider("lancedb")
cognee.config.data_root_directory(str(Path(_DATA_DIR) / "data"))
cognee.config.system_root_directory(str(Path(_DATA_DIR) / "system"))


SAMPLE_TEXT = """\
Acme Corporation 2024 Annual Report.

Acme Corporation reported total revenue of 1.2 billion dollars in 2024,
a 12 percent increase over 2023. The growth was driven primarily by the
Cloud Platform division, which expanded into the European market.

Jane Doe was appointed Chief Executive Officer of Acme Corporation in
March 2024. Under her leadership, operating margin expanded to 18 percent.

Acme Corporation is headquartered in Seattle and employs roughly 4,500 people.
"""


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def main() -> None:
    # Start from a clean slate in the isolated dirs.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Write the sample as a real file so the document name is a clean basename.
    doc_path = Path(_DATA_DIR) / "acme_annual_report_2024.txt"
    doc_path.write_text(SAMPLE_TEXT)

    banner("ADD + COGNIFY")
    await cognee.add(str(doc_path), dataset_name="references_demo")
    await cognee.cognify(datasets=["references_demo"])
    print("Knowledge graph built for: acme_annual_report_2024.txt")

    query = "What were Acme's 2024 revenue and who is the CEO?"

    banner("1) RAG_COMPLETION  (include_references=True)  -> chunk evidence")
    rag = await cognee.search(
        query_text=query,
        query_type=SearchType.RAG_COMPLETION,
        datasets=["references_demo"],
        include_references=True,
    )
    print(rag[0] if rag else "<no result>")

    banner("2) GRAPH_COMPLETION  (include_references=True)  -> graph/entity evidence")
    graph = await cognee.search(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=["references_demo"],
        include_references=True,
    )
    print(graph[0] if graph else "<no result>")

    banner("3) GRAPH_COMPLETION  (include_references=False)  -> no Evidence section")
    plain = await cognee.search(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=["references_demo"],
        include_references=False,
    )
    print(plain[0] if plain else "<no result>")

    answer = plain[0] if plain else ""
    has_evidence = "Evidence:" in (answer if isinstance(answer, str) else str(answer))
    banner("CHECK")
    print(f"include_references=False contains 'Evidence:'  -> {has_evidence} (expected False)")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio

import cognee
from cognee.modules.retrieval.path_search import AdaptivePathSearch, Path
from cognee.shared.logging_utils import ERROR, setup_logging

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"
#
# WARNING: this demo starts by wiping local cognee memory (cognee.forget).


def node_label(node) -> str:
    """Human-readable node label: name when available, else type (e.g. DocumentChunk), else ID."""
    return node.attributes.get("name") or node.attributes.get("type") or node.id


def format_path(path: Path) -> str:
    """Render a path as: A -[rel]-> B <-[rel]- C (arrows follow edge direction)."""
    parts = [node_label(path.nodes[0])]
    current = path.nodes[0]
    for edge, node in zip(path.edges, path.nodes[1:]):
        relationship = edge.attributes.get("relationship_type") or "related_to"
        if edge.node1.id == current.id:
            parts.append(f"-[{relationship}]-> {node_label(node)}")
        else:
            parts.append(f"<-[{relationship}]- {node_label(node)}")
        current = node
    return " ".join(parts)


async def main():
    # Start clean, then remember several separate stories. Entities repeat across
    # the texts (Alice, Acme Corp, Atlas, Nimbus, Berlin/Munich), so the knowledge
    # graph connects facts from different documents — exactly the multi-hop
    # structure that path search is built to surface.
    await cognee.forget(everything=True)
    texts = [
        """
        Alice is a senior engineer at Acme Corp, where she leads the Atlas project.
        Bob collaborates with Alice on the Atlas project and reports to Carol.
        Carol is the head of engineering at Acme Corp.
        """,
        """
        Acme Corp is a logistics company headquartered in Berlin.
        Acme Corp acquired Nimbus Analytics in 2023.
        Nimbus Analytics keeps its main office in Munich.
        """,
        """
        Berlin is the capital of Germany. Munich is a city in Germany.
        Alice moved from Munich to Berlin when she joined Acme Corp.
        """,
        """
        The Atlas project is a route optimization engine written in Rust.
        Nimbus Analytics contributes demand forecasting models to the Atlas project.
        Carol approved the Rust rewrite of the Atlas project.
        """,
    ]
    for text in texts:
        await cognee.remember(text, self_improvement=False)

    query = "How is Alice connected to Munich?"
    print(f"Searching for paths with query: '{query}'\n")

    path_search = AdaptivePathSearch(
        num_seeds=4,
        walks_per_seed=10,
        max_depth=5,
        top_k=5,
        random_seed=42,
    )
    paths = await path_search.run(query)

    if not paths:
        print("No paths found.")
        return

    print(f"Top {len(paths)} paths (of {len(path_search.scored_candidates)} scored candidates):\n")
    for rank, path in enumerate(paths, start=1):
        print(f"{rank}. score={path.score:.3f}  {format_path(path)}")


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())

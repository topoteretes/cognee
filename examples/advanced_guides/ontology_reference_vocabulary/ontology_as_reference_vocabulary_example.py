"""
Ontology as a reference vocabulary for extraction.

The advanced companion to ``examples/guides/ontology_quickstart.py``. That guide applies a
bundled ontology to two inline sentences; this one grounds two bundled real documents
(``data/text_1.txt``, ``data/text_2.txt``) in ``data/basic_ontology.owl``, building the
ontology ``Config`` with ``RDFLibOntologyResolver`` explicitly, then queries the
ontology-constrained graph and renders it for inspection.

Usage:
    uv run python examples/advanced_guides/ontology_reference_vocabulary/ontology_as_reference_vocabulary_example.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

import asyncio
import os
from pathlib import Path

import cognee
from cognee import SearchType, visualize_graph
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.shared.logging_utils import setup_logging

with open(
    os.path.join(Path(__file__).resolve().parent, "data", "text_1.txt"), "r", encoding="utf-8"
) as f:
    text_1 = f.read()

with open(
    os.path.join(Path(__file__).resolve().parent, "data", "text_2.txt"), "r", encoding="utf-8"
) as f:
    text_2 = f.read()


async def main():
    # Step 1: Reset data and system state
    await cognee.forget(everything=True)

    ontology_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "basic_ontology.owl"
    )

    # Create full config structure manually
    config: Config = {
        "ontology_config": {
            "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
        }
    }

    # Step 2: Remember text using the ontology-backed graph config.
    text_list = [text_1, text_2]
    await cognee.remember(text_list, config=config, self_improvement=False)
    print("Knowledge with ontology created.")

    # Step 3: Query insights
    search_results = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What are the exact cars and their types produced by Audi?",
    )
    print(search_results)

    visualize_graph_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "ontology_as_reference_vocabulary.html"
    )
    await visualize_graph(visualize_graph_path)


if __name__ == "__main__":
    logger = setup_logging()
    asyncio.run(main())

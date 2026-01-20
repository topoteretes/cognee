import asyncio
import os

import cognee
from cognee.api.v1.search import SearchType
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.shared.logging_utils import setup_logging
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.ontology_config import Config

with open("data/text_1.txt", "r", encoding="utf-8") as f:
    text_1 = f.read()

with open("data/text_2.txt", "r", encoding="utf-8") as f:
    text_2 = f.read()


async def main():
    # Step 1: Reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Step 2: Add text
    text_list = [text_1, text_2]
    await cognee.add(text_list)

    # Step 3: Create knowledge graph

    ontology_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/basic_ontology.owl"
    )

    # Create full config structure manually
    config: Config = {
        "ontology_config": {
            "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
        }
    }

    await cognee.cognify(config=config)
    print("Knowledge with ontology created.")

    # Step 4: Query insights
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What are the exact cars and their types produced by Audi?",
    )
    print(search_results)

    await visualize_graph()


if __name__ == "__main__":
    logger = setup_logging()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())

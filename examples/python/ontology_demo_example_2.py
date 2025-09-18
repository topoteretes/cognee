import cognee
import asyncio
from cognee.shared.logging_utils import setup_logging
import os
import textwrap
from cognee.api.v1.search import SearchType
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.ontology_config import Config


async def run_pipeline(ontology_path=None):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    scientific_papers_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scientific_papers/"
    )

    await cognee.add(scientific_papers_dir)

    config: Config = {
        "ontology_config": {
            "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
        }
    }

    pipeline_run = await cognee.cognify(config=config)

    return pipeline_run


async def query_pipeline(questions):
    answers = []
    for question in questions:
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text=question,
        )
        answers.append(search_results)

    return answers


def print_comparison_table(questions, answers_with, answers_without, col_width=45):
    separator = "-" * (col_width * 3 + 6)

    header = f"{'Question'.ljust(col_width)} | {'WITH Ontology (owl grounded facts)'.ljust(col_width)} | {'WITHOUT Ontology'.ljust(col_width)}"
    logger.info(separator)
    logger.info(header)
    logger.info(separator)

    for q, with_o, without_o in zip(questions, answers_with, answers_without):
        q_wrapped = textwrap.fill(q, width=col_width)
        with_o_wrapped = textwrap.fill(str(with_o), width=col_width)
        without_o_wrapped = textwrap.fill(str(without_o), width=col_width)

        q_lines = q_wrapped.split("\n")
        with_lines = with_o_wrapped.split("\n")
        without_lines = without_o_wrapped.split("\n")

        max_lines = max(len(q_lines), len(with_lines), len(without_lines))

        for i in range(max_lines):
            q_line = q_lines[i] if i < len(q_lines) else ""
            with_line = with_lines[i] if i < len(with_lines) else ""
            without_line = without_lines[i] if i < len(without_lines) else ""
            logger.info(
                f"{q_line.ljust(col_width)} | {with_line.ljust(col_width)} | {without_line.ljust(col_width)}"
            )

        logger.info(separator)


async def main():
    questions = [
        "What are common risk factors for Type 2 Diabetes?",
        "What preventive measures reduce the risk of Hypertension?",
        "What symptoms indicate possible Cardiovascular Disease?",
        "I have blurred vision and a headache. What diease do I have?",
        "What diseases are associated with Obesity?",
    ]

    ontology_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ontology_input_example/enriched_medical_ontology_with_classes.owl",
    )

    logger.info("\n--- Generating answers WITH ontology ---\n")
    await run_pipeline(ontology_path=ontology_path)
    answers_with_ontology = await query_pipeline(questions)

    logger.info("\n--- Generating answers WITHOUT ontology ---\n")
    await run_pipeline()
    answers_without_ontology = await query_pipeline(questions)

    print_comparison_table(questions, answers_with_ontology, answers_without_ontology)

    await visualize_graph()


if __name__ == "__main__":
    logger = setup_logging()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())

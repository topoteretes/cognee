from dotenv import load_dotenv
import json
import os
from dataclasses import dataclass
from graphrag_sdk.source import URL, STRING
from graphrag_sdk import KnowledgeGraph, Ontology
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from falkordb import FalkorDB

load_dotenv()


def _get_sources_from_corpus_json(corpus_file, limit=None, batch_size=None):
    """Returns STRING sources from corpus JSON file."""
    with open(corpus_file, "r") as file:
        corpus = json.load(file)

    if batch_size:
        # Create batches of entries
        batches = []
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i : i + batch_size]
            batched_text = " ".join(batch)
            batches.append(batched_text)

        sources = [STRING(batch) for batch in batches]
    else:
        # Process individual entries
        sources = [STRING(instance) for instance in corpus]

    return sources[:limit] if limit else sources


def create_ontology(
    corpus_file="hotpot_50_corpus.json",
    output_file="hotpot_qa_ontology.json",
    model_name="gpt-4o-mini",
):
    """Creates ontology from corpus data."""
    print(f"Loading corpus from {corpus_file}...")
    sources = _get_sources_from_corpus_json(corpus_file, batch_size=100)
    print(f"Processing {len(sources)} source documents for ontology detection")

    model = LiteModel(model_name=model_name)
    print(f"Using model: {model_name}")

    # Ontology Auto-Detection
    print("Starting ontology auto-detection...")
    ontology = Ontology.from_sources(
        sources=sources,
        model=model,
    )
    print(
        f"Ontology detection complete with {len(ontology.entities)} entities and {len(ontology.relations)} relations"
    )

    # Save the ontology to the disk as a json file.
    print(f"Saving ontology to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(json.dumps(ontology.to_json(), indent=2))

    return ontology


def create_knowledge_graph(
    ontology_file="hotpot_qa_ontology.json",
    corpus_file="hotpot_50_corpus.json",
    kg_name="hotpot_qa_kg",
    model_name="gpt-4o-mini",
    host="127.0.0.1",
    port=6379,
    username=None,
    password=None,
    recreate=False,
):
    """Creates knowledge graph from ontology file."""
    print(f"Loading ontology from {ontology_file}...")
    with open(ontology_file, "r", encoding="utf-8") as file:
        ontology = Ontology.from_json(json.loads(file.read()))

    print(
        f"Ontology loaded with {len(ontology.entities)} entities and {len(ontology.relations)} relations"
    )
    model = LiteModel(model_name=model_name)

    # Check if graph exists and delete if recreate is True
    db = FalkorDB(host=host, port=port, username=username, password=password)
    available_graphs = db.list_graphs()
    graph_exists = kg_name in available_graphs

    if graph_exists and recreate:
        print(f"Deleting existing knowledge graph '{kg_name}'")
        graph = db.select_graph(kg_name)
        graph.delete()
    elif graph_exists:
        print(f"Knowledge graph '{kg_name}' already exists")

    print(f"Initializing knowledge graph '{kg_name}'...")
    kg = KnowledgeGraph(
        name=kg_name,
        model_config=KnowledgeGraphModelConfig.with_model(model),
        ontology=ontology,
        host=host,
        port=port,
        username=username,
        password=password,
    )

    # Only process sources if we're creating a new graph or recreating
    if recreate or not graph_exists:
        print(f"Loading corpus from {corpus_file}...")
        sources = _get_sources_from_corpus_json(corpus_file, batch_size=10)
        print(f"Processing {len(sources)} instances for knowledge graph...")
        kg.process_sources(sources)
        print("Source processing complete")
    else:
        print("Using existing knowledge graph data - skipping source processing")

    return kg


def answer_questions(
    kg, qa_pairs_file="hotpot_50_qa_pairs.json", print_results=True, output_file=None
):
    """Query knowledge graph with questions from qa pairs file."""
    print(f"Loading QA pairs from {qa_pairs_file}...")
    with open(qa_pairs_file, "r") as file:
        qa_pairs = json.load(file)

    print(f"Processing {len(qa_pairs)} questions...")
    results = []
    chat = kg.chat_session()

    for i, qa_pair in enumerate(qa_pairs[:2]):
        question = qa_pair.get("question")
        expected_answer = qa_pair.get("answer")

        print(f"Processing question {i + 1}/{len(qa_pairs)}: {question}")
        try:
            response = chat.send_message(question)
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            response = "Error: Unable to generate answer due to an internal error."

        result = {"question": question, "answer": response, "golden_answer": expected_answer}

        if print_results:
            print(f"{json.dumps(result, indent=2)}{'-' * 50}\n")

        results.append(result)

    if output_file:
        print(f"Saving results to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)

    return results


def main(config):
    """Main function for HotpotQA knowledge graph pipeline."""
    print("Starting HotpotQA knowledge graph pipeline...")

    # Create ontology if it doesn't exist
    if not os.path.exists(config.ontology_file):
        print(f"Creating new ontology from corpus file: {config.corpus_file}")
        create_ontology(
            corpus_file=config.corpus_file,
            output_file=config.ontology_file,
            model_name=config.ontology_model,
        )
        print(f"Created new ontology and saved to {config.ontology_file}")
    else:
        print(f"Using existing ontology from {config.ontology_file}")

    # Create knowledge graph and process sources if needed
    print(f"Creating knowledge graph: {config.kg_name}")
    kg = create_knowledge_graph(
        ontology_file=config.ontology_file,
        corpus_file=config.corpus_file,
        kg_name=config.kg_name,
        model_name=config.kg_model,
        host=config.kg_host,
        port=config.kg_port,
        recreate=config.kg_recreate,
    )

    # Answer questions
    print(f"Answering questions from {config.qa_pairs_file}...")
    answer_questions(
        kg=kg,
        qa_pairs_file=config.qa_pairs_file,
        print_results=config.print_results,
        output_file=config.results_file,
    )
    print(f"Results saved to {config.results_file}")
    print("Pipeline completed successfully")


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class HotpotQAConfig:
        """Configuration for HotpotQA knowledge graph pipeline."""

        # Ontology parameters
        corpus_file: str = "hotpot_50_corpus.json"
        ontology_file: str = "hotpot_qa_ontology.json"
        ontology_model: str = "gpt-4o-mini"

        # Knowledge graph parameters
        kg_name: str = "hotpot_qa"
        kg_model: str = "gpt-4o-mini"
        kg_host: str = "127.0.0.1"
        kg_port: int = 6379
        kg_recreate: bool = False

        # QA parameters
        qa_pairs_file: str = "hotpot_50_qa_pairs.json"
        results_file: str = "hotpot_qa_results.json"
        print_results: bool = True

    # Create configuration with default values
    config = HotpotQAConfig()
    main(config)

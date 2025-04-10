import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from tqdm import tqdm

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_helpers import search_results_to_context_string

load_dotenv()


async def load_corpus_to_graphiti(
    graphiti: Graphiti, corpus_file: str = "hotpot_50_corpus.json", limit: int = None
):
    """Loads corpus data into graphiti."""
    print(f"Loading corpus from {corpus_file}...")
    with open("../" + corpus_file) as file:
        corpus = json.load(file)

    # Apply limit if specified
    if limit is not None:
        corpus = corpus[:limit]
        print(f"Limited to first {limit} documents")

    print(f"Adding {len(corpus)} documents to graphiti...")
    for i, document in enumerate(tqdm(corpus, desc="Adding documents")):
        await graphiti.add_episode(
            name=f"Document {i + 1}",
            episode_body=str(document),
            source=EpisodeType.text,
            source_description="corpus",
            reference_time=datetime.now(timezone.utc),
        )

    print("All documents added to graphiti")
    return graphiti

  
async def search_graphiti(query: str, graphiti_client: Graphiti) -> str:
    """Search the graphiti graph for information related to the query"""
    results = await graphiti_client.search_(query=query)
    return search_results_to_context_string(results)


async def answer_questions(
    graphiti: Graphiti,
    model_name: str = "gpt-4o-mini",
    qa_pairs_file: str = "hotpot_50_qa_pairs.json",
    print_results: bool = True,
    output_file: str = None,
    limit: int = None,
):
    """Answer questions using graphiti retrieval with direct LLM calls."""
    print(f"Loading QA pairs from {qa_pairs_file}...")
    with open("../" + qa_pairs_file) as file:
        qa_pairs = json.load(file)

    # Apply limit if specified
    if limit is not None:
        qa_pairs = qa_pairs[:limit]
        print(f"Limited to first {limit} questions")

    print(f"Processing {len(qa_pairs)} questions...")
    results = []

    # Set up LLM
    llm = ChatOpenAI(model=model_name, temperature=0)

    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair.get("question")
        expected_answer = qa_pair.get("answer")

        print(f"Processing question {i + 1}/{len(qa_pairs)}: {question}")

        # Get context from graphiti
        context = await search_graphiti(question, graphiti)

        # Create messages with system prompt and context
        messages = [
            {
                "role": "system",
                "content": "Use only the provided context and your reasoning to answer the question. "
                           "Reason through your answer before giving a final ANSWER",
            },
            {"role": "user", "content": f"\n{context}\n\nQuestion: {question}"
                                        "Respond with a JSON in the following format: {\"reasoning\": \"reasoning required to reach an answer\", \"answer\": \"The final answer to the question as a result of the provided reasoning.\"}"},
        ]

        # Get answer from LLM
        response = await llm.ainvoke(messages)
        answer = json.loads(response.content)["answer"]

        # No need to store the question and answer in Graphiti, not storing the pair will make rerunning the eval easier.

        result = {"question": question, "answer": answer, "golden_answer": expected_answer}

        if print_results:
            print(
                f"Question {i + 1}: {question}\nResponse: {answer}\nExpected: {expected_answer}\n{'-' * 50}"
            )

        results.append(result)

    if output_file:
        print(f"Saving results to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)

    return results


async def main_async(config):
    """Main async function for HotpotQA graphiti pipeline."""
    print("Starting HotpotQA graphiti pipeline...")
    print(f"Configuration: {config}")

    # Initialize clients
    graphiti = Graphiti(config.db_url, config.db_user, config.db_password)
    await graphiti.build_indices_and_constraints(delete_existing=True)
    # openai_client = OpenAI()

    # Load corpus to graphiti
    await load_corpus_to_graphiti(
        graphiti=graphiti, corpus_file=config.corpus_file, limit=config.corpus_limit
    )

    # Answer questions
    print(f"Answering questions from {config.qa_pairs_file}...")
    await answer_questions(
        graphiti=graphiti,
        model_name=config.model_name,
        qa_pairs_file=config.qa_pairs_file,
        print_results=config.print_results,
        output_file=config.results_file,
        limit=config.qa_limit,
    )

    await graphiti.close()
    print(f"Results saved to {config.results_file}")
    print("Pipeline completed successfully")


def main(config):
    """Wrapper for async main function."""
    asyncio.run(main_async(config))


if __name__ == "__main__":

    @dataclass
    class HotpotQAGraphitiConfig:
        """Configuration for HotpotQA graphiti pipeline."""

        # Database parameters
        db_url: str = os.getenv("NEO4J_URI")
        db_user: str = os.getenv("NEO4J_USER")
        db_password: str = os.getenv("NEO4J_PASSWORD")

        # Corpus parameters
        corpus_file: str = "hotpot_50_corpus.json"
        corpus_limit: int = None  # Limit number of documents to process

        # Model parameters
        model_name: str = "gpt-4o-mini"

        # QA parameters
        qa_pairs_file: str = "hotpot_50_qa_pairs.json"
        qa_limit: int = None  # Limit number of questions to process
        results_file: str = "hotpot_qa_graphiti_results.json"
        print_results: bool = True

    # Create configuration with default values
    config = HotpotQAGraphitiConfig()
    main(config)

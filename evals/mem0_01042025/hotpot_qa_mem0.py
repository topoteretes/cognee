from dotenv import load_dotenv
import json
from dataclasses import dataclass
from openai import OpenAI
from mem0 import Memory
from tqdm import tqdm

load_dotenv()


def load_corpus_to_memory(
    memory: Memory,
    corpus_file: str = "hotpot_50_corpus.json",
    user_id: str = "hotpot_qa_user",
    limit: int = None,
):
    """Loads corpus data into memory."""
    print(f"Loading corpus from {corpus_file}...")
    with open(corpus_file, "r") as file:
        corpus = json.load(file)

    # Apply limit if specified
    if limit is not None:
        corpus = corpus[:limit]
        print(f"Limited to first {limit} documents")

    print(f"Adding {len(corpus)} documents to memory...")
    for i, document in enumerate(tqdm(corpus, desc="Adding documents")):
        # Create a conversation that includes the document content
        messages = [
            {"role": "system", "content": "This is a document to remember."},
            {"role": "user", "content": "Please remember this document."},
            {"role": "assistant", "content": document},
        ]
        memory.add(messages, user_id=user_id)

    print("All documents added to memory")
    return memory


def answer_questions(
    memory: Memory,
    openai_client: OpenAI,
    model_name: str = "gpt-4o-mini",
    user_id: str = "hotpot_qa_user",
    qa_pairs_file: str = "hotpot_50_qa_pairs.json",
    print_results: bool = True,
    output_file: str = None,
    limit: int = None,
):
    """Answer questions using memory retrieval."""
    print(f"Loading QA pairs from {qa_pairs_file}...")
    with open(qa_pairs_file, "r") as file:
        qa_pairs = json.load(file)

    # Apply limit if specified
    if limit is not None:
        qa_pairs = qa_pairs[:limit]
        print(f"Limited to first {limit} questions")

    print(f"Processing {len(qa_pairs)} questions...")
    results = []

    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair.get("question")
        expected_answer = qa_pair.get("answer")

        print(f"Processing question {i + 1}/{len(qa_pairs)}: {question}")

        # Retrieve relevant memories
        relevant_memories = memory.search(query=question, user_id=user_id, limit=5)
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])

        # Generate response
        system_prompt = f"You are a helpful AI assistant. Answer the question based on the provided context.\n\nContext:\n{memories_str}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        response = openai_client.chat.completions.create(model=model_name, messages=messages)
        answer = response.choices[0].message.content

        # Store the question and answer in memory
        messages.append({"role": "assistant", "content": answer})
        memory.add(messages, user_id=user_id)

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


def main(config):
    """Main function for HotpotQA memory pipeline."""
    print("Starting HotpotQA memory pipeline...")
    print(f"Configuration: {config}")

    # Initialize clients
    memory = Memory()
    openai_client = OpenAI()

    # Load corpus to memory
    load_corpus_to_memory(
        memory=memory,
        corpus_file=config.corpus_file,
        user_id=config.user_id,
        limit=config.corpus_limit,
    )

    # Answer questions
    print(f"Answering questions from {config.qa_pairs_file}...")
    answer_questions(
        memory=memory,
        openai_client=openai_client,
        model_name=config.model_name,
        user_id=config.user_id,
        qa_pairs_file=config.qa_pairs_file,
        print_results=config.print_results,
        output_file=config.results_file,
        limit=config.qa_limit,
    )

    print(f"Results saved to {config.results_file}")
    print("Pipeline completed successfully")


if __name__ == "__main__":

    @dataclass
    class HotpotQAMemoryConfig:
        """Configuration for HotpotQA memory pipeline."""

        # Corpus parameters
        corpus_file: str = "hotpot_50_corpus.json"
        corpus_limit: int = None  # Limit number of documents to process

        # Memory parameters
        user_id: str = "hotpot_qa_user"

        # Model parameters
        model_name: str = "gpt-4o-mini"

        # QA parameters
        qa_pairs_file: str = "hotpot_50_qa_pairs.json"
        qa_limit: int = None  # Limit number of questions to process
        results_file: str = "hotpot_qa_mem0_results.json"
        print_results: bool = True

    # Create configuration with default values
    config = HotpotQAMemoryConfig()
    main(config)

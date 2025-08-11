#!/usr/bin/env python3
"""Run Graphiti QA benchmark."""

from qa.qa_benchmark_graphiti import QABenchmarkGraphiti, GraphitiConfig


def main():
    """Run Graphiti benchmark."""
    config = GraphitiConfig(
        corpus_limit=None,  # Small test
        qa_limit=None,
        print_results=True,
    )

    benchmark = QABenchmarkGraphiti.from_jsons(
        corpus_file="hotpot_qa_24_corpus.json",
        qa_pairs_file="hotpot_qa_24_qa_pairs.json",
        config=config,
    )

    results = benchmark.run()
    return results


if __name__ == "__main__":
    main()

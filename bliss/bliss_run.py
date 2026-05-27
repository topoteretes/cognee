import argparse
import asyncio
import json
import os

# Cognee calls setup_logging() on first package import; set level before that.
os.environ["LOG_LEVEL"] = "ERROR"

from cognee.shared.logging_utils import ERROR, setup_logging

setup_logging(log_level=ERROR)

import cognee  # noqa: E402

CANDIDATES = [
    "Problem A is solved by routing inputs through beta, then gamma, then delta.",
    "Module beta should never be used in any problem.",
]


def _print_json(label: str, value) -> None:
    print(f"=== {label} ===")
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, indent=2))


async def run_retrieval(candidate: str, verbose: bool = True) -> None:
    from bliss_retriever import BlissRetriever

    retriever = BlissRetriever()
    if verbose:
        retrieved = await retriever.get_retrieved_objects(query=candidate)
        _print_json("retrieved objects", retrieved.model_dump())
        context = await retriever.get_context_from_objects(
            query=candidate, retrieved_objects=retrieved
        )
        _print_json("context", context)
        completion = await retriever.get_completion_from_context(
            query=candidate, retrieved_objects=retrieved, context=context
        )
        _print_json("completion", completion[0])
        return

    result = (await retriever.get_completion(candidate))[0]
    _print_json("completion", result)


async def main(verbose: bool = True) -> None:
    from bliss_improve import run_hypothesis_decomposition
    from bliss_remember import ingest_papers

    print("Clearing existing data...")
    await cognee.forget(everything=True)

    print("Remembering papers (extract entities and hypotheses)...")
    await ingest_papers()

    print("Improving graph (decompose hypotheses into premise/conclusion)...")
    await run_hypothesis_decomposition()

    for candidate in CANDIDATES:
        print(f"\nRetrieving and scoring candidate:\n  {candidate}\n")
        await run_retrieval(candidate, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print retrieved objects and context before completion (default: true)",
    )
    args = parser.parse_args()
    asyncio.run(main(verbose=args.verbose))

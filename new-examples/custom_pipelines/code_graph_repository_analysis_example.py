import argparse
import asyncio
import cognee
from cognee import SearchType
from cognee.shared.logging_utils import setup_logging, ERROR

from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline


async def main(repo_path, include_docs):
    run_status = False
    async for run_status in run_code_graph_pipeline(repo_path, include_docs=include_docs):
        run_status = run_status

    # Test CODE search
    search_results = await cognee.search(query_type=SearchType.CODE, query_text="test")
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nSearch results are:\n")
    for result in search_results:
        print(f"{result}\n")

    return run_status


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the repository")
    parser.add_argument(
        "--include_docs",
        type=lambda x: x.lower() in ("true", "1"),
        default=False,
        help="Whether or not to process non-code files",
    )
    parser.add_argument(
        "--time",
        type=lambda x: x.lower() in ("true", "1"),
        default=True,
        help="Whether or not to time the pipeline run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)

    args = parse_args()

    if args.time:
        import time

        start_time = time.time()
        asyncio.run(main(args.repo_path, args.include_docs))
        end_time = time.time()
        print("\n" + "=" * 50)
        print(f"Pipeline Execution Time: {end_time - start_time:.2f} seconds")
        print("=" * 50 + "\n")
    else:
        asyncio.run(main(args.repo_path, args.include_docs))

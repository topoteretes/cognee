import argparse
import asyncio
import logging

from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
from cognee.shared.utils import setup_logging


async def main(repo_path, include_docs):
    return await run_code_graph_pipeline(repo_path, include_docs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the repository")
    parser.add_argument(
        "--include_docs",
        type=lambda x: x.lower() in ("true", "1"),
        default=True,
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
    setup_logging(logging.ERROR)

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

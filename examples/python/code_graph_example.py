import argparse
import asyncio

from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline


async def main(repo_path, include_docs):
    async for result in run_code_graph_pipeline(repo_path, include_docs):
        print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the repository")
    parser.add_argument("--include_docs", type=lambda x: x.lower() in ("true", "1"), default=True, help="Whether or not to process non-code files")
    args = parser.parse_args()
    asyncio.run(main(args.repo_path, args.include_docs))
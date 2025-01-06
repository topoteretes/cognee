import argparse
import asyncio

from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline


async def main(repo_path, include_docs):
    async for result in run_code_graph_pipeline(repo_path, include_docs):
        print(result)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_path", 
        type=str, 
        required=True, 
        help="Path to the repository"
    )
    parser.add_argument(
        "--include_docs",
        type=lambda x: x.lower() in ("true", "1"),
        default=True,
        help="Whether or not to process non-code files"
    )
    parser.add_argument(
        "--mock_embedding",
        type=lambda x: x.lower() in ("true", "1"), 
        default=True,
        help="Whether or not to mock embedding and code summary"
    )
    parser.add_argument(
        "--mock_code_summary",
        type=lambda x: x.lower() in ("true", "1"),
        default=True, 
        help="Whether or not to mock code summary"
    )
    parser.add_argument(
        "--time",
        type=lambda x: x.lower() in ("true", "1"),
        default=True,
        help="Whether or not to time the pipeline run"
    )
    return parser.parse_args()

if __name__ == "__main__":
    import os

    args = parse_args()
    
    if args.mock_embedding:
        os.environ["MOCK_EMBEDDING"] = "true"
        print("Mocking embedding.")
    
    if args.mock_code_summary:
        os.environ["MOCK_CODE_SUMMARY"] = "true"
        print("Mocking code summary.")
    
    if args.time:
        import time
        start_time = time.time()
        asyncio.run(main(args.repo_path, args.include_docs))
        end_time = time.time()
        print("\n" + "="*50)
        print(f"Pipeline Execution Time: {end_time - start_time:.2f} seconds")
        print("="*50 + "\n")
    else:
        asyncio.run(main(args.repo_path, args.include_docs))
        
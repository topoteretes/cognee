import argparse
import asyncio
from evals.eval_swe_bench import run_code_graph_pipeline

async def main(repo_path):
    async for result in await run_code_graph_pipeline(repo_path):
        print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=str, required=True, help="Path to the repository")
    args = parser.parse_args()
    asyncio.run(main(args.repo_path))


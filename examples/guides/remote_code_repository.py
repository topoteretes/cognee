"""Ingest a remote repository through add → scoped cognify, without LLM calls.

Run: uv run python examples/guides/remote_code_repository.py \
    https://github.com/owner/repo --dataset new_code
For a private repository set CODE_REPO_TOKEN. Existing legacy graphs need a new
dataset; this example never deletes data. Requires Git and the bundled Enola.
"""

import argparse
import asyncio
import os

import cognee
from cognee.modules.pipelines.exceptions import PipelineRunFailedError
from cognee.modules.pipelines.models import PipelineRunErrored
from cognee.tasks.code_graph.config import CodeGraphConfig


async def main(repo_url: str, dataset: str):
    options: CodeGraphConfig = {"include_documents": False}
    token = os.getenv("CODE_REPO_TOKEN")
    if token:
        options["repo_credentials"] = token
    added = await cognee.add(repo_url, dataset_name=dataset, codegraph_config=options)
    if isinstance(added, PipelineRunErrored):
        raise PipelineRunFailedError(message=added.error_message or "Repository add failed.")
    if not added.data_ids:
        raise ValueError("Repository add returned no data IDs; refusing an unscoped build.")
    await cognee.cognify(datasets=[added.dataset_id], data_ids=added.data_ids)
    print(f"Indexed repository in dataset {added.dataset_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_url")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.repo_url, args.dataset))

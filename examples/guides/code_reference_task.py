"""Link an ingested README/document/issue to an exact code symbol.

First ingest code with add -> cognify and store the source document's chunks.
Then pass the real dataset, source data item and chunk IDs to this example:

uv run python examples/guides/code_reference_task.py \
    --dataset DATASET_UUID --data-id DATA_UUID --source-node-id CHUNK_UUID \
    --repo github.com/acme/app --file src/orders.py --symbol src/orders.save \
    --source-url https://linear.app/acme/issue/SDK-42 --external-id SDK-42 \
    --evidence 'The task explicitly names src/orders.py save.'

The custom pipeline checks dataset membership and supplies the real source
context. The link task checks chunk ownership. No provider fetch or LLM call.
"""

import argparse
import asyncio
import json
from uuid import UUID

import cognee
from cognee import SearchType
from cognee.modules.data.models import Data
from cognee.modules.pipelines import Task
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.code_graph import CodeReference, link_code_references


async def link_source(data, ctx: PipelineContext, reference: CodeReference):
    """Run after source storage; references come from explicit source evidence."""
    report = await link_code_references([reference], ctx)
    if report["unresolved"]:
        raise ValueError(report["unresolved"])
    print(json.dumps(report, indent=2))
    return report


async def read_links(dataset_id: UUID, symbol_id: str):
    """Use the target_id from the link report to disambiguate same-named symbols."""
    return await cognee.search(
        query_type=SearchType.CODE,
        query_text="",
        datasets=[dataset_id],
        code_query={"operation": "references", "id": symbol_id},
    )


async def main(args):
    reference = CodeReference(
        source_node_id=args.source_node_id,
        repo=args.repo,
        file=args.file,
        symbol=args.symbol,
        source_url=args.source_url,
        external_id=args.external_id,
        evidence=args.evidence,
    )
    # run_custom_pipeline reloads this persisted row under the dataset lock;
    # a data item from another dataset is rejected before the task runs.
    result = await cognee.run_custom_pipeline(
        tasks=[Task(link_source, reference=reference)],
        data=[Data(id=args.data_id)],
        dataset=args.dataset,
        pipeline_name="explicit_code_references",
        skip_connection_test=True,
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "data-id", "source-node-id"):
        parser.add_argument(f"--{name}", type=UUID, required=True)
    for name in ("repo", "file", "symbol", "source-url", "evidence"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--external-id")
    asyncio.run(main(parser.parse_args()))

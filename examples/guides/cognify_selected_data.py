"""Process selected documents of a dataset with cognify(data=...).

Two documents are added to one dataset, and cognify builds the graph for one of them. The
other document stays pending until a later run.

`data` takes Data rows of exactly one dataset. Pass that dataset alone in `datasets`. Rows of
another dataset raise before the run starts.

Requires: LLM_API_KEY.
Run: uv run python examples/guides/cognify_selected_data.py
"""

import asyncio

import cognee

DATASET = "selected_data_demo"


async def main():
    await cognee.forget(everything=True)

    await cognee.add("Alice maintains the billing service.", dataset_name=DATASET)
    await cognee.add("Bob owns the deployment pipeline.", dataset_name=DATASET)

    dataset = next(d for d in await cognee.datasets.list_datasets() if d.name == DATASET)
    rows = await cognee.datasets.list_data(dataset.id)

    await cognee.cognify(datasets=[dataset.id], data=rows[:1])

    rows = await cognee.datasets.list_data(dataset.id)
    done = [row for row in rows if "cognify_pipeline" in (row.pipeline_status or {})]
    print(f"Cognified {len(done)} of {len(rows)} documents")


if __name__ == "__main__":
    asyncio.run(main())

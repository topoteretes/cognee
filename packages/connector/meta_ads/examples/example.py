import asyncio
import os

import cognee
from cognee_community_connector_meta_ads import meta_ads_source


async def main() -> None:
    source = meta_ads_source(
        access_token=os.environ["META_ADS_ACCESS_TOKEN"],
        account_ids=os.environ["META_ADS_ACCOUNT_IDS"].split(","),
        window=os.environ.get("META_ADS_WINDOW", "last_30d"),
    )
    await cognee.remember(
        source,
        dataset_name="meta_ads",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    answer = await cognee.recall(
        query_text="What ad copy and targeting have we used, and what performed best?",
        datasets=["meta_ads"],
    )
    for result in answer:
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())

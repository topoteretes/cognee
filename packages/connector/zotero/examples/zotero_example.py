import os

import cognee
from cognee_community_connector_zotero import zotero_source


async def main():
    await cognee.remember(
        zotero_source(
            api_key=os.environ["ZOTERO_API_KEY"],
            user_id=os.environ["ZOTERO_USER_ID"],
            include_references=True,
            include_notes=True,
            include_attachments=True,
        ),
        dataset_name="zotero_library",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    print(await cognee.search("What papers mention retrieval?", datasets=["zotero_library"]))

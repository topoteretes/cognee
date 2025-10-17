import asyncio

import cognee


async def main():
    await cognee.prune.prune_data()
    print("Data pruned.")

    await cognee.prune.prune_system(metadata=True)

    extraction_rules = {
        "title": {"selector": "title"},
        "headings": {"selector": "h1, h2, h3", "all": True},
        "links": {
            "selector": "a",
            "attr": "href",
            "all": True,
        },
        "paragraphs": {"selector": "p", "all": True},
    }

    loaders_config = {
        "web_url_loader": {
            "soup_config": {
                "max_depth": 1,
                "follow_links": False,
                "extraction_rules": extraction_rules,
            }
        }
    }

    await cognee.add(
        "https://en.wikipedia.org/wiki/Large_language_model",
        preferred_loaders=["web_url_loader"],
        incremental_loading=False,  # TODO: incremental loading bypasses regular data ingestion, which breaks. Will fix
        loaders_config=loaders_config,
    )

    await cognee.cognify()
    print("Knowledge graph created.")

    await cognee.visualize_graph()
    print("Data visualized")


if __name__ == "__main__":
    asyncio.run(main())

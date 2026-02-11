import asyncio
from os import path
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

    await cognee.add(
        "https://en.wikipedia.org/api/rest_v1/page/html/Large_language_model",
        incremental_loading=False,
        preferred_loaders={"beautiful_soup_loader": {"extraction_rules": extraction_rules}},
    )

    await cognee.cognify()
    print("Knowledge graph created.")

    graph_visualization_path = path.join(path.dirname(__file__), "web_url_example.html")
    await cognee.visualize_graph(graph_visualization_path)
    print("Data visualized")


if __name__ == "__main__":
    asyncio.run(main())

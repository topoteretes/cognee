import asyncio
from os import path

import cognee


async def main():
    await cognee.forget(everything=True)
    print("Data forgotten.")

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

    await cognee.remember(
        "https://en.wikipedia.org/api/rest_v1/page/html/Large_language_model",
        incremental_loading=False,
        preferred_loaders={"beautiful_soup_loader": {"extraction_rules": extraction_rules}},
        self_improvement=False,
    )

    print("Knowledge graph created.")

    graph_visualization_path = path.join(
        path.dirname(__file__), ".artifacts", "web_url_example.html"
    )
    await cognee.visualize_graph(graph_visualization_path)


if __name__ == "__main__":
    asyncio.run(main())

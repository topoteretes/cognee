import asyncio
import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from disambiguate_entities import disambiguate_entities_pipeline
from pathlib import Path
import os


async def main(example, custom_prompt, use_poc):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    graph_visualization_path = os.path.join(
        os.path.dirname(__file__),
        f"results/cognify_simple_{example}{'_poc' if use_poc else ''}_graph.html",
    )
    # with open("results/"+example, "w") as f:
    #     print("", file=f)

    with open(
        os.path.join(Path(__file__).resolve().parent, example + ".txt"), "r", encoding="utf-8"
    ) as f:
        text = f.read()
    text = text.split("\n")
    for line in text:
        await cognee.add(line)
        if use_poc:
            await disambiguate_entities_pipeline(custom_prompt=custom_prompt)
        else:
            await cognee.cognify()

    await visualize_graph(graph_visualization_path)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with open("prompts/prompt1.txt", "r") as f:
        custom_prompt_text = f.read()

    try:
        for i in range(1, 2):
            # loop.run_until_complete(main(example="example"+str(i), use_poc=False))
            loop.run_until_complete(
                main(example="example" + str(i), custom_prompt=custom_prompt_text, use_poc=True)
            )
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())

import modal
import os
import logging
import asyncio
import cognee
import sentry_sdk
import concurrent.futures
import signal

from cognee.api.v1.search import SearchType
from cognee.shared.utils import setup_logging

app = modal.App("cognee-runner")

image = (
    modal.Image.from_dockerfile(path="Dockerfile_modal", force_build=False)
    .copy_local_file("pyproject.toml", "pyproject.toml")
    .copy_local_file("poetry.lock", "poetry.lock")
    .env({"ENV": os.getenv("ENV"), "LLM_API_KEY": os.getenv("LLM_API_KEY")})
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
    .pip_install("protobuf", "h2")
)


@app.function(image=image, concurrency_limit=5)
async def entry(text: str, query: str):
    try:
        setup_logging(logging.ERROR)
        sentry_sdk.init(dsn=None)
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await cognee.add(text)
        await cognee.cognify()
        search_results = await cognee.search(SearchType.GRAPH_COMPLETION, query_text=query)

        return {
            "text": text,
            "query": query,
            "answer": search_results[0] if search_results else None,
        }
    finally:
        await asyncio.sleep(1)


@app.local_entrypoint()
async def main():
    text_queries = [
        {
            "text": "The Mars 2023 mission discovered 4.3% water content in soil samples from Jezero Crater.",
            "query": "What percentage of water was found in Jezero Crater's soil based on the provided context?",
        },
        {
            "text": "Bluefin tuna populations decreased by 72% in the Mediterranean between 2010-2022 according to WWF.",
            "query": "What percentage of water was found in Jezero Crater's soil based on the provided context?",
        },
        {
            "text": "Tesla's Q2 2024 report shows 412,000 Model Y vehicles produced with new 4680 battery cells.",
            "query": "How many Model Y cars used the 4680 batteries in Q2 2024?",
        },
        {
            "text": "A 2023 Johns Hopkins study found 23-minute daily naps improve cognitive performance by 18% in adults.",
            "query": "What duration of daily naps boosts cognition according to the 2023 study?",
        },
    ]

    tasks = [entry.remote.aio(item["text"], item["query"]) for item in text_queries]

    results = await asyncio.gather(*tasks)

    print("\nFinal Results:")

    for result in results:
        print(result)
        print("----")

    os.kill(os.getpid(), signal.SIGKILL)

    return 0

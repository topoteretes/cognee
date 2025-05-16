import modal
import os
from cognee.shared.logging_utils import get_logger
import asyncio
import cognee
import signal


from cognee.modules.search.types import SearchType

app = modal.App("cognee-runner")

image = (
    modal.Image.from_dockerfile(path="Dockerfile_modal", force_build=False)
    .copy_local_file("pyproject.toml", "pyproject.toml")
    .copy_local_file("poetry.lock", "poetry.lock")
    .env({"ENV": os.getenv("ENV"), "LLM_API_KEY": os.getenv("LLM_API_KEY")})
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
    .pip_install("protobuf", "h2")
)


@app.function(image=image, concurrency_limit=10)
async def entry(text: str, query: str):
    logger = get_logger()
    logger.info("Initializing Cognee")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(text)
    await cognee.cognify()
    search_results = await cognee.search(query_type=SearchType.GRAPH_COMPLETION, query_text=query)

    return {
        "text": text,
        "query": query,
        "answer": search_results[0] if search_results else None,
    }


@app.local_entrypoint()
async def main():
    logger = get_logger()
    text_queries = [
        {
            "text": "NASA's Artemis program aims to return humans to the Moon by 2026, focusing on sustainable exploration and preparing for future Mars missions.",
            "query": "When does NASA plan to return humans to the Moon under the Artemis program?",
        },
        {
            "text": "According to a 2022 UN report, global food waste amounts to approximately 931 million tons annually, with households contributing 61% of the total.",
            "query": "How much food waste do households contribute annually according to the 2022 UN report?",
        },
        {
            "text": "The 2021 census data revealed that Tokyo's population reached 14 million, reflecting a 2.1% increase compared to the previous census conducted in 2015.",
            "query": "What was Tokyo's population according to the 2021 census data?",
        },
        {
            "text": "A recent study published in the Journal of Nutrition found that consuming 30 grams of almonds daily can lower LDL cholesterol levels by 7% over a 12-week period.",
            "query": "How much can daily almond consumption lower LDL cholesterol according to the study?",
        },
        {
            "text": "Amazon's Prime membership grew to 200 million subscribers in 2023, marking a 10% increase from the previous year, driven by exclusive content and faster delivery options.",
            "query": "How many Prime members did Amazon have in 2023?",
        },
        {
            "text": "A new report by the International Energy Agency states that global renewable energy capacity increased by 295 gigawatts in 2022, primarily driven by solar and wind power expansion.",
            "query": "By how much did global renewable energy capacity increase in 2022 according to the report?",
        },
        {
            "text": "The World Health Organization reported in 2023 that the global life expectancy has risen to 73.4 years, an increase of 5.5 years since the year 2000.",
            "query": "What is the current global life expectancy according to the WHO's 2023 report?",
        },
        {
            "text": "The FIFA World Cup 2022 held in Qatar attracted a record-breaking audience of 5 billion people across various digital and traditional broadcasting platforms.",
            "query": "How many people watched the FIFA World Cup 2022?",
        },
        {
            "text": "The European Space Agency's JUICE mission, launched in 2023, aims to explore Jupiter's icy moons, including Ganymede, Europa, and Callisto, over the next decade.",
            "query": "Which moons is the JUICE mission set to explore?",
        },
        {
            "text": "According to a report by the International Labour Organization, the global unemployment rate in 2023 was estimated at 5.4%, reflecting a slight decrease compared to the previous year.",
            "query": "What was the global unemployment rate in 2023 according to the ILO?",
        },
    ]

    tasks = [entry.remote.aio(item["text"], item["query"]) for item in text_queries]

    results = await asyncio.gather(*tasks)

    logger.info("Final Results:")

    for result in results:
        logger.info(result)
        logger.info("----")

    os.kill(os.getpid(), signal.SIGTERM)

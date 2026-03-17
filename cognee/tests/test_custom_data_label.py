import asyncio
import cognee
from cognee.shared.logging_utils import setup_logging, ERROR
from cognee.api.v1.search import SearchType


async def main():
    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # cognee knowledge graph will be created based on this text
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """
    from cognee.tasks.ingestion.data_item import DataItem

    test_item = DataItem(text, "test_item")
    # Add the text, and make it available for cognify
    await cognee.add(test_item)

    # Use LLMs and cognee to create knowledge graph
    ret_val = await cognee.cognify()

    query_text = "Tell me about NLP"
    print(f"Searching cognee for insights with query: '{query_text}'")
    # Query cognee for insights on the added text
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=query_text
    )

    print("Search results:")
    # Display results
    for result_text in search_results:
        print(result_text)

    from cognee.modules.data.methods.get_dataset_data import get_dataset_data

    for pipeline in ret_val.values():
        dataset_id = pipeline.dataset_id

    dataset_data = await get_dataset_data(dataset_id=dataset_id)

    from fastapi.encoders import jsonable_encoder

    data = [
        dict(
            **jsonable_encoder(data),
            dataset_id=dataset_id,
        )
        for data in dataset_data
    ]

    # Check if label is properly added and stored
    assert data[0]["label"] == "test_item"


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())

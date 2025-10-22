import os
import pathlib
import asyncio
import time

import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def process_and_search(num_of_searches):
    start_time = time.time()

    await cognee.cognify()

    await asyncio.gather(
        *[
            cognee.search(query_text="Tell me about AI", query_type=SearchType.GRAPH_COMPLETION)
            for _ in range(num_of_searches)
        ]
    )

    end_time = time.time()

    return end_time - start_time


async def main():
    file_path = os.path.join(
        pathlib.Path(__file__).resolve().parent, "test_data/artificial-intelligence.pdf"
    )
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_load")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_load")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    num_of_pdfs = 10
    num_of_reps = 5
    upper_boundary_minutes = 10
    average_minutes = 8

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    for i in range(num_of_pdfs):
        await cognee.add(file_path, dataset_name=f"dataset_{i}")

    recorded_times = await asyncio.gather(
        *[process_and_search(num_of_pdfs) for _ in range(num_of_reps)]
    )

    average_recorded_time = sum(recorded_times) / len(recorded_times)

    assert average_recorded_time <= average_minutes * 60

    assert all(rec_time <= upper_boundary_minutes * 60 for rec_time in recorded_times)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import random
import time
from typing import List
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import get_graph_from_model

random.seed(1500)


class Repository(DataPoint):
    path: str
    metadata: dict = {"index_fields": []}


class CodeFile(DataPoint):
    part_of: Repository
    contains: List["CodePart"] = []
    depends_on: List["CodeFile"] = []
    source_code: str
    metadata: dict = {"index_fields": []}


class CodePart(DataPoint):
    part_of: CodeFile
    source_code: str
    metadata: dict = {"index_fields": []}


CodeFile.model_rebuild()
CodePart.model_rebuild()


def nanoseconds_to_largest_unit(nanoseconds):
    # Define conversion factors
    conversion_factors = {
        "weeks": 7 * 24 * 60 * 60 * 1e9,
        "days": 24 * 60 * 60 * 1e9,
        "hours": 60 * 60 * 1e9,
        "minutes": 60 * 1e9,
        "seconds": 1e9,
        "miliseconds": 1e6,
        "microseconds": 1e3,
    }

    # Iterate through conversion factors to find the largest unit
    for unit, factor in conversion_factors.items():
        converted_value = nanoseconds / factor
        if converted_value >= 1:
            return converted_value, unit

    # If nanoseconds is smaller than a second
    return nanoseconds, "nanoseconds"


async def test_circular_reference_extraction():
    repo = Repository(path="repo1")

    code_files = [
        CodeFile(
            id=uuid5(NAMESPACE_OID, f"file{file_index}"),
            source_code="source code",
            part_of=repo,
            contains=[],
            depends_on=[
                CodeFile(
                    id=uuid5(NAMESPACE_OID, f"file{random_id}"),
                    source_code="source code",
                    part_of=repo,
                    depends_on=[],
                )
                for random_id in [random.randint(0, 1499) for _ in range(random.randint(0, 5))]
            ],
        )
        for file_index in range(1500)
    ]

    for code_file in code_files:
        code_file.contains.extend(
            [
                CodePart(
                    part_of=code_file,
                    source_code=f"Part {part_index}",
                )
                for part_index in range(random.randint(1, 20))
            ]
        )

    nodes = []
    edges = []

    added_nodes = {}
    added_edges = {}

    start = time.perf_counter_ns()

    results = await asyncio.gather(
        *[
            get_graph_from_model(code_file, added_nodes=added_nodes, added_edges=added_edges)
            for code_file in code_files
        ]
    )

    time_to_run = time.perf_counter_ns() - start

    print(nanoseconds_to_largest_unit(time_to_run))

    for result_nodes, result_edges in results:
        nodes.extend(result_nodes)
        edges.extend(result_edges)

    assert len(nodes) == 1501
    assert len(edges) == 1501 * 20 + 1500 * 5


if __name__ == "__main__":
    asyncio.run(test_circular_reference_extraction())

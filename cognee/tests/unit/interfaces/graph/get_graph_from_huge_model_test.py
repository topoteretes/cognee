import time
import random
import pytest
import asyncio
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


@pytest.mark.asyncio
async def test_circular_reference_extraction():
    repo = Repository(path="repo1")

    code_files = [
        CodeFile(
            id=uuid5(NAMESPACE_OID, f"file{file_index}"),
            source_code="source code",
            part_of=repo,
            contains=[],
            depends_on=[],
        )
        for file_index in range(1500)
    ]

    for index, code_file in enumerate(code_files):
        first_index = index
        second_index = index

        while first_index == index:
            first_index = random.randint(0, len(code_files) - 1)

        while second_index == index:
            second_index = random.randint(0, len(code_files) - 1)

        code_file.depends_on.extend(
            [
                code_files[first_index],
                code_files[second_index],
            ]
        )
        code_file.contains.extend(
            [
                CodePart(
                    part_of=code_file,
                    source_code=f"Part {part_index}",
                )
                for part_index in range(2)
            ]
        )

    nodes = []
    edges = []

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    start = time.perf_counter_ns()

    results = await asyncio.gather(
        *[
            get_graph_from_model(
                code_file,
                added_nodes=added_nodes,
                added_edges=added_edges,
                visited_properties=visited_properties,
            )
            for code_file in code_files
        ]
    )

    time_to_run = time.perf_counter_ns() - start

    print(nanoseconds_to_largest_unit(time_to_run))

    for result_nodes, result_edges in results:
        nodes.extend(result_nodes)
        edges.extend(result_edges)

    code_files = [node for node in nodes if node.type == "CodeFile"]
    code_parts = [node for node in nodes if node.type == "CodePart"]

    assert len(code_files) == 1500
    assert len(code_parts) == 3000


if __name__ == "__main__":
    asyncio.run(test_circular_reference_extraction())

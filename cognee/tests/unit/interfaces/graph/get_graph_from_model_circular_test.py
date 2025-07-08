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
    part_of: CodeFile
    source_code: str
    metadata: dict = {"index_fields": []}


CodeFile.model_rebuild()
CodePart.model_rebuild()


@pytest.mark.asyncio
async def test_circular_reference_extraction():
    repo = Repository(path="repo1")

    code_file_1 = CodeFile(
        id=uuid5(NAMESPACE_OID, "file_0"),
        source_code="source code",
        part_of=repo,
        contains=[],
        depends_on=[],
    )
    code_part_1 = CodePart(source_code="part_0", part_of=code_file_1)
    code_file_1.contains.append(code_part_1)

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(
        code_file_1,
        added_nodes=added_nodes,
        added_edges=added_edges,
        visited_properties=visited_properties,
    )

    assert len(nodes) == 3
    assert len(edges) == 3


if __name__ == "__main__":
    asyncio.run(test_circular_reference_extraction())

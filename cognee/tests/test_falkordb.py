import os
import pathlib
import sys

# Ensure local cognee is used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cognee
from cognee.infrastructure.files.storage import get_storage_config
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.engine.models import NodeSet
from cognee.context_global_variables import agent_graph_name_ctx

logger = get_logger()

from typing import Optional, List, Union
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import Entity
from cognee.tasks.temporal_graph.models import Event
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.tasks.chunks import chunk_by_paragraph
from cognee.modules.chunking.Chunker import Chunker
from uuid import NAMESPACE_OID, uuid5

class FalkorDBDocumentChunk(DocumentChunk):
    contains: Optional[List[Union[Entity, Event, tuple[Edge, Entity]]]] = None

class FalkorDBTextChunker(Chunker):
    async def read(self):
        paragraph_chunks = []
        async for content_text in self.get_text():
            for chunk_data in chunk_by_paragraph(
                content_text,
                self.max_chunk_size,
                batch_paragraphs=True,
            ):
                if self.chunk_size + chunk_data["chunk_size"] <= self.max_chunk_size:
                    paragraph_chunks.append(chunk_data)
                    self.chunk_size += chunk_data["chunk_size"]
                else:
                    if len(paragraph_chunks) == 0:
                        yield FalkorDBDocumentChunk(
                            id=chunk_data["chunk_id"],
                            text=chunk_data["text"],
                            chunk_size=chunk_data["chunk_size"],
                            is_part_of=self.document,
                            chunk_index=self.chunk_index,
                            cut_type=chunk_data["cut_type"],
                            contains=[],
                            metadata={
                                "index_fields": ["text"],
                            },
                        )
                        paragraph_chunks = []
                        self.chunk_size = 0
                    else:
                        chunk_text = " ".join(chunk["text"] for chunk in paragraph_chunks)
                        try:
                            yield FalkorDBDocumentChunk(
                                id=uuid5(
                                    NAMESPACE_OID, f"{str(self.document.id)}-{self.chunk_index}"
                                ),
                                text=chunk_text,
                                chunk_size=self.chunk_size,
                                is_part_of=self.document,
                                chunk_index=self.chunk_index,
                                cut_type=paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                                contains=[],
                                metadata={
                                    "index_fields": ["text"],
                                },
                            )
                        except Exception as e:
                            logger.error(e)
                            raise e
                        paragraph_chunks = [chunk_data]
                        self.chunk_size = chunk_data["chunk_size"]

                    self.chunk_index += 1

        if len(paragraph_chunks) > 0:
            try:
                yield FalkorDBDocumentChunk(
                    id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{self.chunk_index}"),
                    text=" ".join(chunk["text"] for chunk in paragraph_chunks),
                    chunk_size=self.chunk_size,
                    is_part_of=self.document,
                    chunk_index=self.chunk_index,
                    cut_type=paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                    contains=[],
                    metadata={"index_fields": ["text"]},
                )
            except Exception as e:
                logger.error(e)
                raise e



async def main():
    # 1. Config for FalkorDB
    # Disable internal multi-user check to allow our agent-based isolation
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    
    cognee.config.set_graph_database_provider("falkordb")
    cognee.config.set_vector_db_provider("falkordb")
    
    # Ensuring explicit connection details
    cognee.config.set_graph_db_config({
        "graph_database_url": "localhost",
        "graph_database_port": 6379,
    })
    cognee.config.set_vector_db_config({
        "vector_db_url": "localhost",
        "vector_db_port": 6379,
    })

    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_falkordb")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_falkordb")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "cs_explanations"

    explanation_file_path_nlp = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()

    is_empty = await graph_engine.is_empty()

    assert is_empty, "Graph has to be empty"

    # print("[DEBUG] Adding data with cognee.add...")
    # await cognee.add([explanation_file_path_nlp], dataset_name)
    # print("[DEBUG] cognee.add completed.")

    # explanation_file_path_quantum = os.path.join(
    #     pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt"
    # )

    # print("[DEBUG] Adding second file...")
    # await cognee.add([explanation_file_path_quantum], dataset_name)
    # print("[DEBUG] Second file added.")
    
    # is_empty = await graph_engine.is_empty()
    # assert is_empty, "Graph has to be empty before cognify"

    # print("[DEBUG] Starting cognify (This may take a moment)...")
    # await cognee.cognify([dataset_name], chunker=FalkorDBTextChunker)
    # print("[DEBUG] cognify completed!")

    # is_empty = await graph_engine.is_empty()

    # assert not is_empty, "Graph shouldn't be empty"

    # from cognee.infrastructure.databases.vector import get_vector_engine
    
    # # 2. Search Integration Test
    # vector_engine = get_vector_engine()
    # # Search for "Quantum computer" - should exist
    # results = await vector_engine.search("Entity_name", "Quantum computer")
    # if len(results) > 0:
    #     random_node = results[0]
    #     random_node_name = random_node.payload["text"]

    #     search_results = await cognee.search(
    #         query_type=SearchType.GRAPH_COMPLETION, query_text=random_node_name
    #     )
    #     assert len(search_results) != 0, "The search results list is empty."
    #     print("\n\nExtracted sentences are:\n")
    #     for result in search_results:
    #         print(f"{result}\n")

    #     search_results = await cognee.search(query_type=SearchType.CHUNKS, query_text=random_node_name)
    #     assert len(search_results) != 0, "The search results list is empty."
    #     print("\n\nExtracted chunks are:\n")
    #     for result in search_results:
    #         print(f"{result}\n")

    #     search_results = await cognee.search(
    #         query_type=SearchType.SUMMARIES, query_text=random_node_name
    #     )
    #     assert len(search_results) != 0, "Query related summaries don't exist."
    #     print("\nExtracted results are:\n")
    #     for result in search_results:
    #         print(f"{result}\n")

    # user = await get_default_user()
    # history = await get_history(user.id)

    # Note: History count depends on exact calls; Neo4j test says 6, we'll verify.
    # assert len(history) == 6, "Search history is not correct."

    nodeset_text = "FalkorDB is a graph database that supports cypher."

    await cognee.add([nodeset_text], dataset_name, node_set=["first"])

    await cognee.cognify([dataset_name], chunker=FalkorDBTextChunker)

    # DEBUG: List all nodes to verify NodeSet existence
    graph_client = await get_graph_engine()
    nodes_res = await graph_client.query("MATCH (n) RETURN n")
    print("\n[DEBUG] All Nodes in Graph:")
    for r in nodes_res:
         n = r.get("n") or r.get(0)
         props = n.properties if hasattr(n, "properties") else (n if isinstance(n, dict) else {})
         print(f"Node: labels={getattr(n, 'labels', 'N/A')} props={props}")
    print("[DEBUG] End of Node List\n")
    
    edges_res = await graph_client.query("MATCH (a)-[r]->(b) RETURN a, type(r), b")
    print("\n[DEBUG] All Edges in Graph:")
    for r in edges_res:
         a = r.get("a")
         b = r.get("b")
         rel_type = r.get("type(r)") or r.get(1)
         print(f"Edge: {a.properties.get('id', 'N/A')} -[{rel_type}]-> {b.properties.get('id', 'N/A')}")
    print("[DEBUG] End of Edge List\n")

    context_nonempty = await GraphCompletionRetriever(
        node_type=NodeSet,
        node_name=["first"],
    ).get_context("What is in the context?")

    context_empty = await GraphCompletionRetriever(
        node_type=NodeSet,
        node_name=["nonexistent"],
    ).get_context("What is in the context?")

    assert isinstance(context_nonempty, list) and context_nonempty != [], (
        f"Nodeset_search_test:Expected non-empty string for context_nonempty, got: {context_nonempty!r}"
    )

    assert context_empty == [], (
        f"Nodeset_search_test:Expected empty string for context_empty, got: {context_empty!r}"
    )

    # --- NEW: Multi-Agent Isolation Test ---
    print("\n-------------------------------------------------------------")
    print("TESTING MULTI-AGENT ISOLATION")
    print("-------------------------------------------------------------")
    
    # 1. Agent Red Context
    print("Switching to AgentRed...")
    agent_graph_name_ctx.set("AgentRed")
    # For isolation, we use a distinct dataset name to be safe in Cognee's logic too
    await cognee.add(["Apple is a red fruit."], "red_dataset")
    await cognee.cognify(["red_dataset"], chunker=FalkorDBTextChunker)

    
    # Verify Red can see it
    red_results = await cognee.search("Apple", SearchType.CHUNKS)
    assert len(red_results) > 0, "AgentRed should find 'Apple'"
    
    # 2. Agent Blue Context
    print("Switching to AgentBlue...")
    token_blue = agent_graph_name_ctx.set("AgentBlue") # Override context
    
    # Verify Blue CANNOT see Red's data
    print("AgentBlue searching for 'Apple' (should be empty)...")
    blue_leak_results = await cognee.search("Apple", SearchType.CHUNKS)
    
    # Filter strictly for leakage
    leak_found = False
    for res in blue_leak_results:
        # Check text depending on result object structure
        result_str = str(res).lower()
        if "red fruit" in result_str:
            leak_found = True
            
    if leak_found:
        print("LEAK DETECTED: AgentBlue found AgentRed's data")
        # Ensure we catch this failure
        assert False, "Multi-agent isolation failed!"
    else:
        print("SUCCESS: AgentBlue did not see AgentRed's data.")
        
    print("Multi-Agent Isolation Test: PASSED")
    # ---------------------------------------

    await cognee.prune.prune_data()
    data_root_directory = get_storage_config()["data_root_directory"]
    assert not os.path.isdir(data_root_directory), "Local data files are not deleted"

    await cognee.prune.prune_system(metadata=True)
    is_empty = await graph_engine.is_empty()
    assert is_empty, "FalkorDB graph database is not empty"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

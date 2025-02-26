import asyncio
import aiofiles

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from typing import List, Dict, Any
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt


class CodeQueryInfo(BaseModel):
    """Response model for information extraction from the query"""

    filenames: List[str] = []
    sourcecode: str


async def code_graph_retrieval(query: str) -> list[dict[str, Any]]:
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")

    file_name_collections = ["CodeFile_name"]
    classes_and_functions_collections = [
        "ClassDefinition_source_code",
        "FunctionDefinition_source_code",
    ]

    try:
        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()
    except Exception as e:
        raise RuntimeError("Database initialization error in code_graph_retriever, ") from e

    system_prompt = read_query_prompt("codegraph_retriever_system.txt")

    llm_client = get_llm_client()
    try:
        files_and_codeparts = await llm_client.acreate_structured_output(
            text_input=query,
            system_prompt=system_prompt,
            response_model=CodeQueryInfo,
        )
    except Exception as e:
        raise RuntimeError("Failed to retrieve structured output from LLM") from e

    similar_filenames = []
    similar_codepieces = []

    if not files_and_codeparts.filenames or not files_and_codeparts.sourcecode:
        for collection in file_name_collections:
            search_results_file = await vector_engine.search(collection, query, limit=3)
            for res in search_results_file:
                similar_filenames.append({"id": res.id, "score": res.score, "payload": res.payload})

        for collection in classes_and_functions_collections:
            search_results_code = await vector_engine.search(collection, query, limit=3)
            for res in search_results_code:
                similar_codepieces.append(
                    {"id": res.id, "score": res.score, "payload": res.payload}
                )

    else:
        for collection in file_name_collections:
            for file_from_query in files_and_codeparts.filenames:
                search_results_file = await vector_engine.search(
                    collection, file_from_query, limit=3
                )
                for res in search_results_file:
                    similar_filenames.append(
                        {"id": res.id, "score": res.score, "payload": res.payload}
                    )

        for collection in classes_and_functions_collections:
            for code_from_query in files_and_codeparts.sourcecode:
                search_results_code = await vector_engine.search(
                    collection, code_from_query, limit=3
                )
                for res in search_results_code:
                    similar_codepieces.append(
                        {"id": res.id, "score": res.score, "payload": res.payload}
                    )

    file_ids = [str(item["id"]) for item in similar_filenames]
    code_ids = [str(item["id"]) for item in similar_codepieces]

    relevant_triplets = await asyncio.gather(
        *[graph_engine.get_connections(node_id) for node_id in code_ids + file_ids]
    )

    paths = set()

    for sublist in relevant_triplets:
        for tpl in sublist:
            if isinstance(tpl, tuple) and len(tpl) >= 3:
                if "file_path" in tpl[0]:
                    paths.add(tpl[0]["file_path"])
                if "file_path" in tpl[2]:  # Third tuple element
                    paths.add(tpl[2]["file_path"])

    retrieved_files = {}

    read_tasks = []
    for file_path in paths:

        async def read_file(fp):
            try:
                async with aiofiles.open(fp, "r", encoding="utf-8") as f:
                    retrieved_files[fp] = await f.read()
            except Exception as e:
                print(f"Error reading {fp}: {e}")
                retrieved_files[fp] = ""

        read_tasks.append(read_file(file_path))

    await asyncio.gather(*read_tasks)

    result = [
        {
            "name": file_path,
            "description": file_path,
            "content": retrieved_files[file_path],
        }
        for file_path in paths
    ]

    return result

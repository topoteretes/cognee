from typing import Any, Optional, List
import asyncio
import aiofiles
from pydantic import BaseModel

from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt


class CodeRetriever(BaseRetriever):
    """Retriever for handling code-based searches."""

    class CodeQueryInfo(BaseModel):
        """Response model for information extraction from the query"""

        filenames: List[str] = []
        sourcecode: str

    def __init__(self, top_k: int = 3):
        """Initialize retriever with search parameters."""
        self.top_k = top_k
        self.file_name_collections = ["CodeFile_name"]
        self.classes_and_functions_collections = [
            "ClassDefinition_source_code",
            "FunctionDefinition_source_code",
        ]

    async def _process_query(self, query: str) -> "CodeRetriever.CodeQueryInfo":
        """Process the query using LLM to extract file names and source code parts."""
        system_prompt = read_query_prompt("codegraph_retriever_system.txt")
        llm_client = get_llm_client()
        try:
            return await llm_client.acreate_structured_output(
                text_input=query,
                system_prompt=system_prompt,
                response_model=self.CodeQueryInfo,
            )
        except Exception as e:
            raise RuntimeError("Failed to retrieve structured output from LLM") from e

    async def get_context(self, query: str) -> Any:
        """Find relevant code files based on the query."""
        if not query or not isinstance(query, str):
            raise ValueError("The query must be a non-empty string.")

        try:
            vector_engine = get_vector_engine()
            graph_engine = await get_graph_engine()
        except Exception as e:
            raise RuntimeError("Database initialization error in code_graph_retriever, ") from e

        files_and_codeparts = await self._process_query(query)

        similar_filenames = []
        similar_codepieces = []

        if not files_and_codeparts.filenames or not files_and_codeparts.sourcecode:
            for collection in self.file_name_collections:
                search_results_file = await vector_engine.search(
                    collection, query, limit=self.top_k
                )
                for res in search_results_file:
                    similar_filenames.append(
                        {"id": res.id, "score": res.score, "payload": res.payload}
                    )

            for collection in self.classes_and_functions_collections:
                search_results_code = await vector_engine.search(
                    collection, query, limit=self.top_k
                )
                for res in search_results_code:
                    similar_codepieces.append(
                        {"id": res.id, "score": res.score, "payload": res.payload}
                    )
        else:
            for collection in self.file_name_collections:
                for file_from_query in files_and_codeparts.filenames:
                    search_results_file = await vector_engine.search(
                        collection, file_from_query, limit=self.top_k
                    )
                    for res in search_results_file:
                        similar_filenames.append(
                            {"id": res.id, "score": res.score, "payload": res.payload}
                        )

            for collection in self.classes_and_functions_collections:
                search_results_code = await vector_engine.search(
                    collection, files_and_codeparts.sourcecode, limit=self.top_k
                )
                for res in search_results_code:
                    similar_codepieces.append(
                        {"id": res.id, "score": res.score, "payload": res.payload}
                    )

        relevant_triplets = await asyncio.gather(
            *[
                graph_engine.get_connections(similar_piece["id"])
                for similar_piece in similar_filenames + similar_codepieces
            ]
        )

        paths = set()
        for sublist in relevant_triplets:
            for tpl in sublist:
                if isinstance(tpl, tuple) and len(tpl) >= 3:
                    if "file_path" in tpl[0]:
                        paths.add(tpl[0]["file_path"])
                    if "file_path" in tpl[2]:
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

        return [
            {
                "name": file_path,
                "description": file_path,
                "content": retrieved_files[file_path],
            }
            for file_path in paths
        ]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Returns the code files context."""
        if context is None:
            context = await self.get_context(query)
        return context

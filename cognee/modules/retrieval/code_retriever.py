from typing import Any, Optional, List
import asyncio
import aiofiles
from pydantic import BaseModel

from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.LLMGateway import LLMGateway

logger = get_logger("CodeRetriever")


class CodeRetriever(BaseRetriever):
    """Retriever for handling code-based searches."""

    class CodeQueryInfo(BaseModel):
        """
        Model for representing the result of a query related to code files.

        This class holds a list of filenames and the corresponding source code extracted from a
        query. It is used to encapsulate response data in a structured format.
        """

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
        logger.debug(
            f"Processing query with LLM: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        system_prompt = read_query_prompt("codegraph_retriever_system.txt")

        try:
            result = await LLMGateway.acreate_structured_output(
                text_input=query,
                system_prompt=system_prompt,
                response_model=self.CodeQueryInfo,
            )
            logger.info(
                f"LLM extracted {len(result.filenames)} filenames and {len(result.sourcecode)} chars of source code"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve structured output from LLM: {str(e)}")
            raise RuntimeError("Failed to retrieve structured output from LLM") from e

    async def get_context(self, query: str) -> Any:
        """Find relevant code files based on the query."""
        logger.info(
            f"Starting code retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        if not query or not isinstance(query, str):
            logger.error("Invalid query: must be a non-empty string")
            raise ValueError("The query must be a non-empty string.")

        try:
            vector_engine = get_vector_engine()
            graph_engine = await get_graph_engine()
            logger.debug("Successfully initialized vector and graph engines")
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise RuntimeError("Database initialization error in code_graph_retriever, ") from e

        files_and_codeparts = await self._process_query(query)

        similar_filenames = []
        similar_codepieces = []

        if not files_and_codeparts.filenames or not files_and_codeparts.sourcecode:
            logger.info("No specific files/code extracted from query, performing general search")

            for collection in self.file_name_collections:
                logger.debug(f"Searching {collection} collection with general query")
                search_results_file = await vector_engine.search(
                    collection, query, limit=self.top_k
                )
                logger.debug(f"Found {len(search_results_file)} results in {collection}")
                for res in search_results_file:
                    similar_filenames.append(
                        {"id": res.id, "score": res.score, "payload": res.payload}
                    )

            existing_collection = []
            for collection in self.classes_and_functions_collections:
                if await vector_engine.has_collection(collection):
                    existing_collection.append(collection)

            if not existing_collection:
                raise RuntimeError("No collection found for code retriever")

            for collection in existing_collection:
                logger.debug(f"Searching {collection} collection with general query")
                search_results_code = await vector_engine.search(
                    collection, query, limit=self.top_k
                )
                logger.debug(f"Found {len(search_results_code)} results in {collection}")
                for res in search_results_code:
                    similar_codepieces.append(
                        {"id": res.id, "score": res.score, "payload": res.payload}
                    )
        else:
            logger.info(
                f"Using extracted filenames ({len(files_and_codeparts.filenames)}) and source code for targeted search"
            )

            for collection in self.file_name_collections:
                for file_from_query in files_and_codeparts.filenames:
                    logger.debug(f"Searching {collection} for specific file: {file_from_query}")
                    search_results_file = await vector_engine.search(
                        collection, file_from_query, limit=self.top_k
                    )
                    logger.debug(
                        f"Found {len(search_results_file)} results for file {file_from_query}"
                    )
                    for res in search_results_file:
                        similar_filenames.append(
                            {"id": res.id, "score": res.score, "payload": res.payload}
                        )

            for collection in self.classes_and_functions_collections:
                logger.debug(f"Searching {collection} with extracted source code")
                search_results_code = await vector_engine.search(
                    collection, files_and_codeparts.sourcecode, limit=self.top_k
                )
                logger.debug(f"Found {len(search_results_code)} results for source code search")
                for res in search_results_code:
                    similar_codepieces.append(
                        {"id": res.id, "score": res.score, "payload": res.payload}
                    )

        total_items = len(similar_filenames) + len(similar_codepieces)
        logger.info(
            f"Total search results: {total_items} items ({len(similar_filenames)} filenames, {len(similar_codepieces)} code pieces)"
        )

        if total_items == 0:
            logger.warning("No search results found, returning empty list")
            return []

        logger.debug("Getting graph connections for all search results")
        relevant_triplets = await asyncio.gather(
            *[
                graph_engine.get_connections(similar_piece["id"])
                for similar_piece in similar_filenames + similar_codepieces
            ]
        )
        logger.info(f"Retrieved graph connections for {len(relevant_triplets)} items")

        paths = set()
        for i, sublist in enumerate(relevant_triplets):
            logger.debug(f"Processing connections for item {i}: {len(sublist)} connections")
            for tpl in sublist:
                if isinstance(tpl, tuple) and len(tpl) >= 3:
                    if "file_path" in tpl[0]:
                        paths.add(tpl[0]["file_path"])
                    if "file_path" in tpl[2]:
                        paths.add(tpl[2]["file_path"])

        logger.info(f"Found {len(paths)} unique file paths to read")

        retrieved_files = {}
        read_tasks = []
        for file_path in paths:

            async def read_file(fp):
                try:
                    logger.debug(f"Reading file: {fp}")
                    async with aiofiles.open(fp, "r", encoding="utf-8") as f:
                        content = await f.read()
                        retrieved_files[fp] = content
                        logger.debug(f"Successfully read {len(content)} characters from {fp}")
                except Exception as e:
                    logger.error(f"Error reading {fp}: {e}")
                    retrieved_files[fp] = ""

            read_tasks.append(read_file(file_path))

        await asyncio.gather(*read_tasks)
        logger.info(
            f"Successfully read {len([f for f in retrieved_files.values() if f])} files (out of {len(paths)} total)"
        )

        result = [
            {
                "name": file_path,
                "description": file_path,
                "content": retrieved_files[file_path],
            }
            for file_path in paths
        ]

        logger.info(f"Returning {len(result)} code file contexts")
        return result

    async def get_completion(
        self, query: str, context: Optional[Any] = None, session_id: Optional[str] = None
    ) -> Any:
        """
        Returns the code files context.

        Parameters:
        -----------

            - query (str): The query string to retrieve code context for.
            - context (Optional[Any]): Optional pre-fetched context; if None, it retrieves
              the context for the query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)

        Returns:
        --------

            - Any: The code files context, either provided or retrieved.
        """
        if context is None:
            context = await self.get_context(query)
        return context

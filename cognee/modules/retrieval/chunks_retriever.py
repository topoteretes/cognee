from typing import Any, Optional, List, Union
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.utils import normalize_graph_result
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError
from datetime import datetime, timezone

logger = get_logger("ChunksRetriever")


class ChunksRetriever(BaseRetriever):
    """
    Handles document chunk-based searches by retrieving relevant chunks and generating
    completions from them.

    Public methods:

    - get_context: Retrieves document chunks based on a query.
    - get_completion: Generates a completion using provided context or retrieves context if
    not given.
    """

    def __init__(
        self,
        top_k: Optional[int] = 5,
        strict_enrichment: bool = False,
    ):
        self.top_k = top_k
        self.strict_enrichment = strict_enrichment

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
        """
        Generates a completion using document chunks context.
        In case of the Chunks Retriever, we do not generate a completion, we just return
        the payloads of found chunks.

        Parameters:
        -----------

            - query (str): The query string to be used for generating a completion.
            - retrieved_objects (Any): The retrieved objects to be used for generating a completion.
            - context (Any): The context to be used for generating a completion.

        Returns:
        --------

            - List[dict]: A list of payloads of found chunks.
        """
        # TODO: Do we want to generate a completion using LLM here?
        if retrieved_objects:
            chunk_payloads = [found_chunk.payload for found_chunk in retrieved_objects]
            return chunk_payloads
        else:
            return []

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Retrieves context from retrieved chunks, in text form.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant document chunks.
            - retrieved_objects (Any): The retrieved objects to be used for generating textual context.

        Returns:
        --------

            - str: A string containing the combined text of the retrieved chunks, or an
              empty string if none are found.
        """
        if retrieved_objects:
            chunk_payload_texts = [found_chunk.payload["text"] for found_chunk in retrieved_objects]
            return "\n".join(chunk_payload_texts)
        else:
            return ""

    async def get_retrieved_objects(self, query: str) -> Any:
        """
        Retrieves document chunks context based on the query.
        Searches for document chunks relevant to the specified query using a vector engine.
        Raises a NoDataError if no data is found in the system.
        Parameters:
        -----------
            - query (str): The query string to search for relevant document chunks.
        Returns:
        --------
            - Any: A list of document chunks retrieved from the search.
        """
        logger.info(
            f"Starting chunk retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        vector_engine = get_vector_engine()

        try:
            found_chunks = await vector_engine.search(
                "DocumentChunk_text", query, limit=self.top_k, include_payload=True
            )
            logger.info(f"Found {len(found_chunks)} chunks from vector search")
        except CollectionNotFoundError as error:
            logger.error("DocumentChunk_text collection not found in vector database")
            raise NoDataError("No data found in the system, please add data first.") from error

        if not found_chunks:
            return found_chunks

        try:
            await update_node_access_timestamps(found_chunks)
        except Exception as error:
            logger.warning(f"Failed to update timestamps: {error}")
            if self.strict_enrichment:
                raise NoDataError(f"Failed to update timestamps: {error}") from error

        # Get graph engine (stores chunk-document relationships)
        try:
            graph_engine = await get_graph_engine()
        except Exception as error:
            error_msg = f"Graph engine unavailable: {error}"
            if self.strict_enrichment:
                logger.error(error_msg)
                raise NoDataError(error_msg) from error
            else:
                logger.warning(f"{error_msg}, skipping enrichment")
                return found_chunks

        chunk_ids = []
        chunk_id_map = {}

        # Extract all chunk IDs for lookup
        for chunk in found_chunks:
            chunk_id = None
            try:
                if hasattr(chunk, "id"):
                    chunk_id = str(chunk.id)
                elif hasattr(chunk, "payload") and "id" in chunk.payload:
                    chunk_id = str(chunk.payload["id"])

                if chunk_id:
                    chunk_ids.append(chunk_id)
                    chunk_id_map[chunk_id] = chunk
                elif self.strict_enrichment:
                    raise ValueError(f"Chunk missing ID: {chunk}")
            except Exception as error:
                if self.strict_enrichment:
                    raise NoDataError(f"Failed to extract chunk ID: {error}") from error
                logger.debug(f"Failed to extract chunk ID: {error}")

        if not chunk_ids:
            logger.warning("No valid chunk IDs found, skipping enrichment")
            return found_chunks

        # Try batched query
        parent_map = {}

        try:
            logger.debug(f"Attempting batched parent lookup for {len(chunk_ids)} chunks")

            cypher_query = """
            MATCH (chunk:DocumentChunk)-[:is_part_of]->(doc:Document)
            WHERE chunk.id IN $chunk_ids
            RETURN chunk.id as chunk_id, doc.id as doc_id, doc.name as doc_name, doc.type as doc_type
            """

            result = await graph_engine.query(cypher_query, params={"chunk_ids": chunk_ids})
            result = normalize_graph_result(result, ["chunk_id", "doc_id", "doc_name", "doc_type"])

            for row in result:
                try:
                    chunk_id = str(row["chunk_id"])
                    parent_map[chunk_id] = {
                        "id": str(row.get("doc_id", "")),
                        "name": row.get("doc_name", "Unknown"),
                    }
                    if "doc_type" in row and row["doc_type"]:
                        parent_map[chunk_id]["type"] = row["doc_type"]
                except Exception as error:
                    logger.warning(f"Failed to parse batch result row: {error}")

            logger.info(
                f"Batched query found parents for {len(parent_map)}/{len(chunk_ids)} chunks"
            )

        except Exception as error:
            # fallback to individual queries
            logger.warning(f"Batched lookup failed, falling back to individual queries: {error}")

            for chunk_id in chunk_ids:
                try:
                    cypher_query = """
                    MATCH (chunk:DocumentChunk {id: $chunk_id})-[:is_part_of]->(doc:Document)
                    RETURN doc.id as doc_id, doc.name as doc_name, doc.type as doc_type
                    LIMIT 1
                    """

                    result = await graph_engine.query(cypher_query, params={"chunk_id": chunk_id})
                    result = normalize_graph_result(result, ["doc_id", "doc_name", "doc_type"])

                    if result and len(result) > 0:
                        row = result[0]
                        parent_map[chunk_id] = {
                            "id": str(row.get("doc_id", "")),
                            "name": row.get("doc_name", "Unknown"),
                        }
                        if "doc_type" in row and row["doc_type"]:
                            parent_map[chunk_id]["type"] = row["doc_type"]

                except Exception as individual_error:
                    if self.strict_enrichment:
                        raise NoDataError(
                            f"Failed to fetch parent for {chunk_id}: {individual_error}"
                        ) from individual_error
                    logger.debug(f"Individual query failed for {chunk_id}: {individual_error}")

        # modify chunk payloads
        enriched_count = 0
        for chunk_id, chunk in chunk_id_map.items():
            if chunk_id in parent_map:
                try:
                    parent_info = parent_map[chunk_id]

                    if hasattr(chunk, "payload") and isinstance(chunk.payload, dict):
                        chunk.payload["parent_document"] = parent_info
                    elif isinstance(chunk, dict):
                        chunk["parent_document"] = parent_info
                    enriched_count += 1
                except Exception as error:
                    if self.strict_enrichment:
                        raise NoDataError(f"Failed to add parent info: {error}") from error
                    logger.warning(f"Failed to add parent info to chunk: {error}")
            elif self.strict_enrichment:
                raise NoDataError(f"No parent found for chunk {chunk_id}")

        success_rate = (enriched_count / len(found_chunks) * 100) if found_chunks else 0
        logger.info(f"Enriched {enriched_count}/{len(found_chunks)} chunks ({success_rate:.1f}%)")

        # vector and graph db out of sync
        if success_rate < 50 and len(found_chunks) > 0 and not self.strict_enrichment:
            logger.warning(
                f"Low enrichment rate ({success_rate:.1f}%) suggests vector/graph DB inconsistency. "
                "Consider running cognee.prune() and re-cognifying."
            )

        return found_chunks

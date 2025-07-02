# PROPOSED TO BE DEPRECATED

"""This module contains the OntologyEngine class which is responsible for adding graph ontology from a JSON or CSV file."""

import csv
import json
from cognee.shared.logging_utils import get_logger
from datetime import datetime, timezone
from fastapi import status
from typing import Any, Dict, List, Optional, Union, Type

import aiofiles
import pandas as pd
from pydantic import BaseModel

from cognee.modules.graph.exceptions import EntityNotFoundError
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.data.chunking.config import get_chunk_config
from cognee.infrastructure.data.chunking.get_chunking_engine import get_chunk_engine
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.files.utils.extract_text_from_file import extract_text_from_file
from cognee.infrastructure.files.utils.guess_file_type import guess_file_type, FileTypeException
from cognee.modules.data.extraction.knowledge_graph.add_model_class_to_graph import (
    add_model_class_to_graph,
)
from cognee.tasks.graph.models import NodeModel, GraphOntology
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.engine.utils import generate_node_id, generate_node_name

logger = get_logger("task:infer_data_ontology")


async def extract_ontology(content: str, response_model: Type[BaseModel]):
    """
    Extracts structured ontology from the provided content using a pre-defined LLM client.

    This asynchronous function retrieves a system prompt from a file and utilizes an LLM
    client to create a structured output based on the input content and specified response
    model.

    Parameters:
    -----------

        - content (str): The content from which to extract the ontology.
        - response_model (Type[BaseModel]): The model that defines the structure of the
          output ontology.

    Returns:
    --------

        The structured ontology extracted from the content.
    """
    llm_client = get_llm_client()

    system_prompt = read_query_prompt("extract_ontology.txt")

    ontology = await llm_client.acreate_structured_output(content, system_prompt, response_model)

    return ontology


class OntologyEngine:
    """
    Manage ontology data and operations for graph structures, providing methods for data
    loading, flattening models, and adding ontological relationships to a graph database.

    Public methods:

    - flatten_model
    - recursive_flatten
    - load_data
    - add_graph_ontology
    """

    async def flatten_model(
        self, model: NodeModel, parent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Flatten the model to a dictionary including optional parent ID and relationship details
        if available.

        Parameters:
        -----------

            - model (NodeModel): The NodeModel instance to flatten.
            - parent_id (Optional[str]): An optional ID of the parent node for hierarchical
              purposes. (default None)

        Returns:
        --------

            - Dict[str, Any]: A dictionary representation of the model with flattened
              attributes.
        """
        result = model.dict()
        result["parent_id"] = parent_id
        if model.default_relationship:
            result.update(
                {
                    "relationship_type": model.default_relationship.type,
                    "relationship_source": model.default_relationship.source,
                    "relationship_target": model.default_relationship.target,
                }
            )
        return result

    async def recursive_flatten(
        self, items: Union[List[Dict[str, Any]], Dict[str, Any]], parent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively flatten a hierarchical structure of models into a flat list of dictionaries.

        Parameters:
        -----------

            - items (Union[List[Dict[str, Any]], Dict[str, Any]]): A list or dictionary
              containing models to flatten.
            - parent_id (Optional[str]): An optional ID of the parent node to maintain hierarchy
              during flattening. (default None)

        Returns:
        --------

            - List[Dict[str, Any]]: A flat list of dictionaries representing the hierarchical
              model structure.
        """
        flat_list = []

        if isinstance(items, list):
            for item in items:
                flat_list.extend(await self.recursive_flatten(item, parent_id))
        elif isinstance(items, dict):
            model = NodeModel.model_validate(items)
            flat_list.append(await self.flatten_model(model, parent_id))
            for child in model.children:
                flat_list.extend(await self.recursive_flatten(child, model.node_id))
        return flat_list

    async def load_data(self, file_path: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load data from a specified JSON or CSV file and return it in a structured format.

        Parameters:
        -----------

            - file_path (str): The path to the file to load data from.

        Returns:
        --------

            - Union[List[Dict[str, Any]], Dict[str, Any]]: Parsed data from the file as either a
              list of dictionaries or a single dictionary depending on content type.
        """
        try:
            if file_path.endswith(".json"):
                async with aiofiles.open(file_path, mode="r") as f:
                    data = await f.read()
                    return json.loads(data)
            elif file_path.endswith(".csv"):
                async with aiofiles.open(file_path, mode="r") as f:
                    content = await f.read()
                    reader = csv.DictReader(content.splitlines())
                    return list(reader)
            else:
                raise IngestionError(message="Unsupported file format")
        except Exception as e:
            raise IngestionError(
                message=f"Failed to load data from {file_path}: {e}",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

    async def add_graph_ontology(self, file_path: str = None, documents: list = None):
        """
        Add graph ontology from a JSON or CSV file, or infer relationships from provided
        document content. Raise exceptions for invalid file types or missing entities.

        Parameters:
        -----------

            - file_path (str): Optional path to a file containing data to be loaded. (default
              None)
            - documents (list): Optional list of document objects for content extraction if no
              file path is provided. (default None)
        """
        if file_path is None:
            initial_chunks_and_ids = []

            chunk_config = get_chunk_config()
            chunk_engine = get_chunk_engine()
            chunk_strategy = chunk_config.chunk_strategy

            for base_file in documents:
                with open(base_file.raw_data_location, "rb") as file:
                    try:
                        file_type = guess_file_type(file)
                        text = extract_text_from_file(file, file_type)

                        subchunks, chunks_with_ids = chunk_engine.chunk_data(
                            chunk_strategy,
                            text,
                            chunk_config.chunk_size,
                            chunk_config.chunk_overlap,
                        )

                        if chunks_with_ids[0][0] == 1:
                            initial_chunks_and_ids.append({base_file.id: chunks_with_ids})

                    except FileTypeException:
                        logger.warning(
                            "File (%s) has an unknown file type. We are skipping it.", file["id"]
                        )

            ontology = await extract_ontology(str(initial_chunks_and_ids), GraphOntology)
            graph_client = await get_graph_engine()

            await graph_client.add_nodes(
                [
                    (
                        node.id,
                        dict(
                            uuid=generate_node_id(node.id),
                            name=generate_node_name(node.name),
                            type=generate_node_id(node.id),
                            description=node.description,
                            updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
                    for node in ontology.nodes
                ]
            )

            await graph_client.add_edges(
                (
                    generate_node_id(edge.source_id),
                    generate_node_id(edge.target_id),
                    edge.relationship_type,
                    dict(
                        source_node_id=generate_node_id(edge.source_id),
                        target_node_id=generate_node_id(edge.target_id),
                        relationship_name=edge.relationship_type,
                        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                for edge in ontology.edges
            )

        else:
            dataset_level_information = documents[0][1]

            # Extract the list of valid IDs from the explanations
            valid_ids = {item["id"] for item in dataset_level_information}
            try:
                data = await self.load_data(file_path)
                flt_ontology = await self.recursive_flatten(data)
                df = pd.DataFrame(flt_ontology)
                graph_client = await get_graph_engine()

                for _, row in df.iterrows():
                    node_data = row.to_dict()
                    node_id = node_data.pop("node_id", None)
                    if node_id in valid_ids:
                        await graph_client.add_node(node_id, node_data)
                    if node_id not in valid_ids:
                        raise EntityNotFoundError(
                            message=f"Node ID {node_id} not found in the dataset"
                        )
                    if pd.notna(row.get("relationship_source")) and pd.notna(
                        row.get("relationship_target")
                    ):
                        await graph_client.add_edge(
                            row["relationship_source"],
                            row["relationship_target"],
                            relationship_name=row["relationship_type"],
                            edge_properties={
                                "source_node_id": row["relationship_source"],
                                "target_node_id": row["relationship_target"],
                                "relationship_name": row["relationship_type"],
                                "updated_at": datetime.now(timezone.utc).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            },
                        )

                return
            except Exception as e:
                raise RuntimeError(f"Failed to add graph ontology from {file_path}: {e}") from e


async def infer_data_ontology(documents, ontology_model=KnowledgeGraph, root_node_id=None):
    """
    Infer data ontology from provided documents and optionally add it to a graph.

    Parameters:
    -----------

        - documents: The documents from which to infer the ontology.
        - ontology_model: The ontology model to use for the inference, defaults to
          KnowledgeGraph. (default KnowledgeGraph)
        - root_node_id: An optional root node identifier for the ontology. (default None)
    """
    if ontology_model == KnowledgeGraph:
        ontology_engine = OntologyEngine()
        root_node_id = await ontology_engine.add_graph_ontology(documents=documents)
    else:
        graph_engine = await get_graph_engine()
        await add_model_class_to_graph(ontology_model, graph_engine)

    yield (documents, root_node_id)

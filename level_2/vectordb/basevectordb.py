# Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client
import logging
from io import BytesIO

from level_2.vectordb.vectordb import PineconeVectorDB, WeaviateVectorDB

logging.basicConfig(level=logging.INFO)
import marvin
import requests
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import WeaviateHybridSearchRetriever
from weaviate.gql.get import HybridFusion

load_dotenv()
from typing import Optional

import tracemalloc

tracemalloc.start()

import os
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document
import uuid
import weaviate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")

LTM_MEMORY_ID_DEFAULT = "00000"
ST_MEMORY_ID_DEFAULT = "0000"
BUFFER_ID_DEFAULT = "0000"


class VectorDBFactory:
    def create_vector_db(
        self,
        user_id: str,
        index_name: str,
        memory_id: str,
        ltm_memory_id: str = LTM_MEMORY_ID_DEFAULT,
        st_memory_id: str = ST_MEMORY_ID_DEFAULT,
        buffer_id: str = BUFFER_ID_DEFAULT,
        db_type: str = "pinecone",
        namespace: str = None,
    ):
        db_map = {"pinecone": PineconeVectorDB, "weaviate": WeaviateVectorDB}

        if db_type in db_map:
            return db_map[db_type](
                user_id,
                index_name,
                memory_id,
                ltm_memory_id,
                st_memory_id,
                buffer_id,
                namespace,
            )

        raise ValueError(f"Unsupported database type: {db_type}")






class BaseMemory:
    def __init__(
        self,
        user_id: str,
        memory_id: Optional[str],
        index_name: Optional[str],
        db_type: str,
        namespace: str,
    ):
        self.user_id = user_id
        self.memory_id = memory_id
        self.index_name = index_name
        self.namespace = namespace
        self.memory_type_id = str(uuid.uuid4())
        self.db_type = db_type
        factory = VectorDBFactory()
        self.vector_db = factory.create_vector_db(
            self.user_id,
            self.index_name,
            self.memory_id,
            db_type=self.db_type,
            namespace=self.namespace,
        )

    def init_client(self, namespace: str):

        return self.vector_db.init_weaviate_client(namespace)

    async def add_memories(
        self,
        observation: Optional[str] = None,
        loader_settings: dict = None,
        params: Optional[dict] = None,
        namespace: Optional[str] = None,
        custom_fields: Optional[str] = None,

    ):

        return await self.vector_db.add_memories(
            observation=observation, loader_settings=loader_settings,
            params=params, namespace=namespace, custom_fields=custom_fields
        )
        # Add other db_type conditions if necessary

    async def fetch_memories(
        self,
        observation: str,
        params: Optional[str] = None,
        namespace: Optional[str] = None,
        n_of_observations: Optional[int] = 2,
    ):

        return await self.vector_db.fetch_memories(
            observation=observation, params=params,
            namespace=namespace,
            n_of_observations=n_of_observations
        )

    async def delete_memories(self, params: Optional[str] = None):
        return await self.vector_db.delete_memories(params)

    # Additional methods for specific Memory can be added here

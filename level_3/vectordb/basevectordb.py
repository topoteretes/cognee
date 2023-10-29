
import logging
from io import BytesIO
import os, sys
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vectordb.vectordb import PineconeVectorDB, WeaviateVectorDB
import sqlalchemy as sa
logging.basicConfig(level=logging.INFO)
import marvin
import requests
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import WeaviateHybridSearchRetriever
from weaviate.gql.get import HybridFusion
from models.sessions import Session
from models.testset import TestSet
from models.testoutput import TestOutput
from models.metadatas import MetaDatas
from models.operation import Operation
from sqlalchemy.orm import sessionmaker
from database.database import engine
load_dotenv()
from typing import Optional
import time
import tracemalloc

tracemalloc.start()

from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.schema import Document
import uuid
import weaviate
from marshmallow import Schema, fields
import json


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")

class VectorDBFactory:
    def __init__(self):
        self.db_map = {
            "pinecone": PineconeVectorDB,
            "weaviate": WeaviateVectorDB,
            # Add more database types and their corresponding classes here
        }

    def create_vector_db(
        self,
        user_id: str,
        index_name: str,
        memory_id: str,
        db_type: str = "weaviate",
        namespace: str = None,
        embeddings=None,
    ):
        if db_type in self.db_map:
            return self.db_map[db_type](
                user_id,
                index_name,
                memory_id,
                namespace,
                embeddings
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
        embeddings: Optional[None],
    ):
        self.user_id = user_id
        self.memory_id = memory_id
        self.index_name = index_name
        self.namespace = namespace
        self.embeddings = embeddings
        self.db_type = db_type
        factory = VectorDBFactory()
        self.vector_db = factory.create_vector_db(
            self.user_id,
            self.index_name,
            self.memory_id,
            db_type=self.db_type,
            namespace=self.namespace,
            embeddings=self.embeddings
        )

    def init_client(self, embeddings, namespace: str):
        return self.vector_db.init_client(embeddings, namespace)


# class VectorDBFactory:
#     def create_vector_db(
#         self,
#         user_id: str,
#         index_name: str,
#         memory_id: str,
#         db_type: str = "pinecone",
#         namespace: str = None,
#         embeddings = None,
#     ):
#         db_map = {"pinecone": PineconeVectorDB, "weaviate": WeaviateVectorDB}
#
#         if db_type in db_map:
#             return db_map[db_type](
#                 user_id,
#                 index_name,
#                 memory_id,
#                 namespace,
#                 embeddings
#             )
#
#         raise ValueError(f"Unsupported database type: {db_type}")
#
# class BaseMemory:
#     def __init__(
#         self,
#         user_id: str,
#         memory_id: Optional[str],
#         index_name: Optional[str],
#         db_type: str,
#         namespace: str,
#         embeddings: Optional[None],
#     ):
#         self.user_id = user_id
#         self.memory_id = memory_id
#         self.index_name = index_name
#         self.namespace = namespace
#         self.embeddings = embeddings
#         self.db_type = db_type
#         factory = VectorDBFactory()
#         self.vector_db = factory.create_vector_db(
#             self.user_id,
#             self.index_name,
#             self.memory_id,
#             db_type=self.db_type,
#             namespace=self.namespace,
#             embeddings=self.embeddings
#         )
#
#     def init_client(self, embeddings, namespace: str):
#
#         return self.vector_db.init_weaviate_client(embeddings, namespace)

    def create_field(self, field_type, **kwargs):
        field_mapping = {
            "Str": fields.Str,
            "Int": fields.Int,
            "Float": fields.Float,
            "Bool": fields.Bool,

        }
        return field_mapping[field_type](**kwargs)

    def create_dynamic_schema(self, params):
        """Create a dynamic schema based on provided parameters."""

        dynamic_fields = {field_name: fields.Str() for field_name in params.keys()}
        # Create a Schema instance with the dynamic fields
        dynamic_schema_instance = Schema.from_dict(dynamic_fields)()
        return dynamic_schema_instance


    async def get_version_from_db(self, user_id, memory_id):
        # Logic to retrieve the version from the database.

        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            # Querying both fields: contract_metadata and created_at
            result = (
                session.query(MetaDatas.contract_metadata, MetaDatas.created_at)
                .filter_by(user_id=user_id)  # using parameter, not self.user_id
                .order_by(MetaDatas.created_at.desc())
                .first()
            )

            if result:

                version_in_db, created_at = result
                logging.info(f"version_in_db: {version_in_db}")
                from ast import literal_eval
                version_in_db= literal_eval(version_in_db)
                version_in_db = version_in_db.get("version")
                return [version_in_db, created_at]
            else:
                return None

        finally:
            session.close()

    async def update_metadata(self, user_id, memory_id, version_in_params, params):
        version_from_db = await self.get_version_from_db(user_id, memory_id)
        Session = sessionmaker(bind=engine)
        session = Session()

        # If there is no metadata, insert it.
        if version_from_db is None:

            session.add(MetaDatas(id = str(uuid.uuid4()), user_id=self.user_id, version = str(int(time.time())) ,memory_id=self.memory_id, contract_metadata=params))
            session.commit()
            return params

        # If params version is higher, update the metadata.
        elif version_in_params > version_from_db[0]:
            session.add(MetaDatas(id = str(uuid.uuid4()), user_id=self.user_id, memory_id=self.memory_id, contract_metadata=params))
            session.commit()
            return params
        else:
            return params


    async def add_memories(
        self,
        observation: Optional[str] = None,
        loader_settings: dict = None,
        params: Optional[dict] = None,
        namespace: Optional[str] = None,
        custom_fields: Optional[str] = None,
        embeddings: Optional[str] = None,

    ):
        from ast import literal_eval
        class DynamicSchema(Schema):
            pass

        default_version = 'current_timestamp'
        version_in_params = params.get("version", default_version)

        # Check and update metadata version in DB.
        schema_fields = params

        def create_field(field_type, **kwargs):
            field_mapping = {
                "Str": fields.Str,
                "Int": fields.Int,
                "Float": fields.Float,
                "Bool": fields.Bool,
            }
            return field_mapping[field_type](**kwargs)

        # Dynamic Schema Creation
        params['user_id'] = self.user_id


        schema_instance = self.create_dynamic_schema(params)  # Always creating Str field, adjust as needed

        logging.info(f"params : {params}")

        # Schema Validation
        schema_instance = schema_instance
        print("Schema fields: ", [field for field in schema_instance._declared_fields])
        loaded_params = schema_instance.load(params)

        return await self.vector_db.add_memories(
            observation=observation, loader_settings=loader_settings,
            params=loaded_params, namespace=namespace, metadata_schema_class = schema_instance, embeddings=embeddings
        )
        # Add other db_type conditions if necessary

    async def fetch_memories(
        self,
        observation: str,
        search_type: Optional[str] = None,
        params: Optional[str] = None,
        namespace: Optional[str] = None,
        n_of_observations: Optional[int] = 2,
    ):
        logging.info(namespace)
        logging.info("The search type is %", search_type)
        logging.info(params)
        logging.info(observation)

        return await self.vector_db.fetch_memories(
            observation=observation, search_type= search_type, params=params,
            namespace=namespace,
            n_of_observations=n_of_observations
        )

    async def delete_memories(self, namespace:str, params: Optional[str] = None):
        return await self.vector_db.delete_memories(namespace,params)


    async def count_memories(self, namespace:str, params: Optional[str] = None):
        return await self.vector_db.count_memories(namespace,params)




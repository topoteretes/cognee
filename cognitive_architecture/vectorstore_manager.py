import logging

from sqlalchemy.future import select

logging.basicConfig(level=logging.INFO)

import os

print(os.getcwd())




from cognitive_architecture.database.postgres.models.user import User
from cognitive_architecture.database.postgres.models.memory import MemoryModel

import ast
import tracemalloc
from cognitive_architecture.database.postgres.database_crud import add_entity

tracemalloc.start()

import os
from dotenv import load_dotenv
import uuid

load_dotenv()



from cognitive_architecture.database.vectordb.basevectordb import BaseMemory

from cognitive_architecture.config import Config

config = Config()
config.load()




class DynamicBaseMemory(BaseMemory):
    def __init__(
        self,
        name: str,
        user_id: str,
        memory_id: str,
        index_name: str,
        db_type: str,
        namespace: str,
        embeddings=None,
    ):
        super().__init__(user_id, memory_id, index_name, db_type, namespace, embeddings)
        self.name = name
        self.attributes = set()
        self.methods = set()
        self.inheritance = None
        self.associations = []

    async def add_method(self, method_name):
        """
        Add a method to the memory class.

        Args:
        - method_name (str): The name of the method to be added.

        Returns:
        None
        """
        self.methods.add(method_name)

    async def add_attribute(self, attribute_name):
        """
        Add an attribute to the memory class.

        Args:
        - attribute_name (str): The name of the attribute to be added.

        Returns:
        None
        """
        self.attributes.add(attribute_name)

    async def get_attribute(self, attribute_name):
        """
        Check if the attribute is in the memory class.

        Args:
        - attribute_name (str): The name of the attribute to be checked.

        Returns:
        bool: True if attribute exists, False otherwise.
        """
        return attribute_name in self.attributes

    async def add_association(self, associated_memory):
        """
        Add an association to another memory class.

        Args:
        - associated_memory (MemoryClass): The memory class to be associated with.

        Returns:
        None
        """
        if associated_memory not in self.associations:
            self.associations.append(associated_memory)
            # Optionally, establish a bidirectional association
            associated_memory.associations.append(self)


class Attribute:
    def __init__(self, name):
        """
        Initialize the Attribute class.

        Args:
        - name (str): The name of the attribute.

        Attributes:
        - name (str): Stores the name of the attribute.
        """
        self.name = name


class Method:
    def __init__(self, name):
        """
        Initialize the Method class.

        Args:
        - name (str): The name of the method.

        Attributes:
        - name (str): Stores the name of the method.
        """
        self.name = name


class Memory:
    def __init__(
        self,
        user_id: str = "676",
        session=None,
        index_name: str = None,
        db_type: str = "weaviate",
        namespace: str = None,
        memory_id: str = None,
        memory_class = None,
        job_id:str = None
    ) -> None:
        self.load_environment_variables()
        self.memory_id = memory_id
        self.user_id = user_id
        self.session = session
        self.index_name = index_name
        self.db_type = db_type
        self.namespace = namespace
        self.memory_instances = []
        self.memory_class = memory_class
        self.job_id=job_id
        # self.memory_class = DynamicBaseMemory(
        #     "Memory", user_id, str(self.memory_id), index_name, db_type, namespace
        # )



    def load_environment_variables(self) -> None:
        load_dotenv()
        self.OPENAI_TEMPERATURE = config.openai_temperature
        self.OPENAI_API_KEY = config.openai_key

    @classmethod
    async def create_memory(cls, user_id: str,  session, job_id:str=None, memory_label:str=None, **kwargs):
        """
        Class method that acts as a factory method for creating Memory instances.
        It performs necessary DB checks or updates before instance creation.
        """
        existing_user = await cls.check_existing_user(user_id, session)
        logging.info(f"Existing user: {existing_user}")

        if existing_user:
            # Handle existing user scenario...
            memory_id = await cls.check_existing_memory(user_id,memory_label, session)
            if memory_id is None:
                memory_id = await cls.handle_new_memory(user_id = user_id, session= session,job_id=job_id, memory_name= memory_label)
            logging.info(
                f"Existing user {user_id} found in the DB. Memory ID: {memory_id}"
            )
        else:
            # Handle new user scenario...
            await cls.handle_new_user(user_id, session)

            memory_id = await cls.handle_new_memory(user_id =user_id, session=session, job_id=job_id, memory_name= memory_label)
            logging.info(
                f"New user {user_id} created in the DB. Memory ID: {memory_id}"
            )

        memory_class = DynamicBaseMemory(
            memory_label, user_id, str(memory_id), index_name=memory_label , db_type='weaviate', **kwargs
        )

        return cls(user_id=user_id, session=session, memory_id=memory_id, job_id =job_id, memory_class=memory_class, **kwargs)

    async def list_memory_classes(self):
        """
        Lists all available memory classes in the memory instance.
        """
        # Use a list comprehension to filter attributes that end with '_class'
        return [attr for attr in dir(self) if attr.endswith("_class")]

    @staticmethod
    async def check_existing_user(user_id: str, session):
        """Check if a user exists in the DB and return it."""
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def check_existing_memory(user_id: str, memory_label:str, session):
        """Check if a user memory exists in the DB and return it. Filters by user and label"""
        try:
            result = await session.execute(
                select(MemoryModel.id).where(MemoryModel.user_id == user_id)
                .filter_by(memory_name=memory_label)
                .order_by(MemoryModel.created_at)

            )
            return result.scalar_one_or_none()
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return None
    @staticmethod
    async def handle_new_user(user_id: str, session):
        """
        Handle new user creation in the database.

        Args:
            user_id (str): The unique identifier for the new user.
            session: The database session for the operation.

        Returns:
            str: A success message or an error message.

        Raises:
            Exception: If any error occurs during the user creation process.
        """
        try:
            new_user = User(id=user_id)
            await add_entity(session, new_user)
            return "User creation successful."
        except Exception as e:
            return f"Error creating user: {str(e)}"

    @staticmethod
    async def handle_new_memory(user_id: str, session, job_id: str = None, memory_name: str = None, memory_category:str='PUBLIC'):
        """
        Handle new memory creation associated with a user.

        Args:
            user_id (str): The user's unique identifier.
            session: The database session for the operation.
            job_id (str, optional): The identifier of the associated job, if any.
            memory_name (str, optional): The name of the memory.

        Returns:
            str: The unique memory ID if successful, or an error message.

        Raises:
            Exception: If any error occurs during memory creation.
        """
        try:
            memory_id = str(uuid.uuid4())
            logging.info("Job id %s", job_id)
            memory = MemoryModel(
                id=memory_id,
                user_id=user_id,
                operation_id=job_id,
                memory_name=memory_name,
                memory_category=memory_category,
                methods_list=str(["Memory", "SemanticMemory", "EpisodicMemory"]),
                attributes_list=str(
                    [
                        "user_id",
                        "index_name",
                        "db_type",
                        "knowledge_source",
                        "knowledge_type",
                        "memory_id",
                        "long_term_memory",
                        "short_term_memory",
                        "namespace",
                    ]
                ),
            )
            await add_entity(session, memory)
            return memory_id
        except Exception as e:
            return f"Error creating memory: {str(e)}"


    async def add_memory_instance(self, memory_class_name: str):
        """Add a new memory instance to the memory_instances list."""
        instance = DynamicBaseMemory(
            memory_class_name,
            self.user_id,
            self.memory_id,
            self.index_name,
            self.db_type,
            self.namespace,
        )
        print("The following instance was defined", instance)
        self.memory_instances.append(instance)

    async def query_method(self):
        methods_list = await self.session.execute(
            select(MemoryModel.methods_list).where(MemoryModel.id == self.memory_id)
        )
        methods_list = methods_list.scalar_one_or_none()
        return methods_list

    async def manage_memory_attributes(self, existing_user):
        """Manage memory attributes based on the user existence."""
        if existing_user:
            print(f"ID before query: {self.memory_id}, type: {type(self.memory_id)}")

            # attributes_list = await self.session.query(MemoryModel.attributes_list).filter_by(id=self.memory_id[0]).scalar()
            attributes_list = await self.query_method()
            logging.info(f"Attributes list: {attributes_list}")
            if attributes_list is not None:
                attributes_list = ast.literal_eval(attributes_list)
                await self.handle_attributes(attributes_list)
            else:
                logging.warning("attributes_list is None!")
        else:
            attributes_list = [
                "user_id",
                "index_name",
                "db_type",
                "knowledge_source",
                "knowledge_type",
                "memory_id",
                "long_term_memory",
                "short_term_memory",
                "namespace",
            ]
            await self.handle_attributes(attributes_list)

    async def handle_attributes(self, attributes_list):
        """Handle attributes for existing memory instances."""
        for attr in attributes_list:
            await self.memory_class.add_attribute(attr)

    async def manage_memory_methods(self, existing_user):
        """
        Manage memory methods based on the user existence.
        """
        if existing_user:
            # Fetch existing methods from the database
            # methods_list = await self.session.query(MemoryModel.methods_list).filter_by(id=self.memory_id).scalar()

            methods_list = await self.session.execute(
                select(MemoryModel.methods_list).where(
                    MemoryModel.id == self.memory_id[0]
                )
            )
            methods_list = methods_list.scalar_one_or_none()
            methods_list = ast.literal_eval(methods_list)
        else:
            # Define default methods for a new user
            methods_list = [
                "async_create_long_term_memory",
                "async_init",
                "add_memories",
                "fetch_memories",
                "delete_memories",
                "async_create_short_term_memory",
                "_create_buffer_context",
                "_get_task_list",
                "_run_main_buffer",
                "_available_operations",
                "_provide_feedback",
            ]
        # Apply methods to memory instances
        for class_instance in self.memory_instances:
            for method in methods_list:
                class_instance.add_method(method)

    async def dynamic_method_call(
        self, dynamic_base_memory_instance, method_name: str, *args, **kwargs
    ):
        if method_name in dynamic_base_memory_instance.methods:
            method = getattr(dynamic_base_memory_instance, method_name, None)
            if method:
                return await method(*args, **kwargs)
        raise AttributeError(
            f"{dynamic_base_memory_instance.name} object has no attribute {method_name}"
        )

    async def add_dynamic_memory_class(self, class_name: str, namespace: str):
        logging.info("Here is the memory id %s", self.memory_id[0])
        new_memory_class = DynamicBaseMemory(
            class_name,
            self.user_id,
            self.memory_id[0],
            self.index_name,
            self.db_type,
            namespace,
        )
        setattr(self, f"{class_name.lower()}_class", new_memory_class)
        return new_memory_class

    async def add_attribute_to_class(self, class_instance, attribute_name: str):
        # add this to database  for a particular user  and load under memory id
        await class_instance.add_attribute(attribute_name)

    async def add_method_to_class(self, class_instance, method_name: str):
        # add this to database for a particular user and load under memory id
        await class_instance.add_method(method_name)


async def main():
    # if you want to run the script as a standalone script, do so with the examples below
    # memory = Memory(user_id="TestUser")
    # await memory.async_init()
    params = {
        "version": "1.0",
        "agreement_id": "AG123456",
        "privacy_policy": "https://example.com/privacy",
        "terms_of_service": "https://example.com/terms",
        "format": "json",
        "schema_version": "1.1",
        "checksum": "a1b2c3d4e5f6",
        "owner": "John Doe",
        "license": "MIT",
        "validity_start": "2023-08-01",
        "validity_end": "2024-07-31",
    }
    loader_settings = {
        "format": "PDF",
        "source": "URL",
        "path": ["https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"],
    }
    # memory_instance = Memory(namespace='SEMANTICMEMORY')
    # sss = await memory_instance.dynamic_method_call(memory_instance.semantic_memory_class, 'fetch_memories', observation='some_observation')

    from database.postgres.database_crud import session_scope
    from database.postgres.database import AsyncSessionLocal

    async with session_scope(AsyncSessionLocal()) as session:
        memory = await Memory.create_memory("677", session, "SEMANTICMEMORY", namespace="SEMANTICMEMORY")
        ff = memory.memory_instances
        logging.info("ssss %s", ff)

        # Adding a memory instance
        # await memory.add_memory_instance("ExampleMemory")

        # Managing memory attributes
        existing_user = await Memory.check_existing_user("677", session)
        print("here is the existing user", existing_user)
        await memory.manage_memory_attributes(existing_user)
        # aeehuvyq_semanticememory_class

        await memory.add_dynamic_memory_class("semanticmemory", "SEMANTICMEMORY")
        await memory.add_method_to_class(memory.semanticmemory_class, "add_memories")
        await memory.add_method_to_class(memory.semanticmemory_class, "fetch_memories")
        sss = await memory.dynamic_method_call(memory.semanticmemory_class, 'add_memories',
                                                        observation='some_observation', params=params, loader_settings=loader_settings)

        # susu = await memory.dynamic_method_call(
        #     memory.semanticmemory_class,
        #     "fetch_memories",
        #     observation="document summary",
        # )
        # print(susu)

    # Adding a dynamic memory class
    # dynamic_memory = memory.add_dynamic_memory_class("DynamicMemory", "ExampleNamespace")

    # memory_instance = Memory(namespace='PROCEDURALMEMORY', session=session)
    # procedural_memory_class = memory_instance.add_dynamic_memory_class('ProceduralMemory', 'PROCEDURALMEMORY')
    # memory_instance.add_method_to_class(procedural_memory_class, 'add_memories')
    #

    # print(sss)
    # load_jack_london = await memory._add_semantic_memory(observation = "bla", loader_settings=loader_settings, params=params)
    # print(load_jack_london)

    modulator = {"relevance": 0.1, "frequency": 0.1}


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

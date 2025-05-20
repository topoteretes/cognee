from .vector_db_interface import VectorDBInterface
from .supported_adapters import supported_adapters


def use_vector_adapter(vector_adapter_name: str, vector_adapter: VectorDBInterface):
    supported_adapters[vector_adapter_name] = vector_adapter

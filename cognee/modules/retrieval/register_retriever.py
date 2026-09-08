from typing import Type

from ..search.types import SearchType
from .base_retriever import BaseRetriever
from .registered_community_retrievers import registered_community_retrievers


def use_retriever(search_type: SearchType, retriever: type[BaseRetriever]):
    """Register a retriever class for a given search type."""
    registered_community_retrievers[search_type] = retriever

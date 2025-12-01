from typing import Type

from .base_retriever import BaseRetriever
from .registered_community_retrievers import registered_community_retrievers
from ..search.types import SearchType


def use_retriever(search_type: SearchType, retriever: Type[BaseRetriever]):
    """Register a retriever class for a given search type."""
    registered_community_retrievers[search_type] = retriever

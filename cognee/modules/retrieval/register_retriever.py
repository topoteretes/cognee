from .base_retriever import BaseRetriever
from .registered_community_retrievers import registered_community_retrievers
from ..search.types import SearchType


def register_retriever(search_type: SearchType, retriever: BaseRetriever):
    registered_community_retrievers[search_type] = retriever

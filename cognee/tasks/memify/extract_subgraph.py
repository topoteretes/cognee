from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment


async def extract_subgraph(subgraphs):
    for subgraph in subgraphs:
        for edge in subgraph:
            yield edge

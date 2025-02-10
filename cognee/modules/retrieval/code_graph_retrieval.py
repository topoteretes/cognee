from cognee.modules.retrieval.description_to_codepart_search import (
    code_description_to_code_part_search,
)


async def code_graph_retrieval(query, include_docs=False):
    retrieved_codeparts, __ = await code_description_to_code_part_search(
        query, include_docs=include_docs
    )

    return [
        {
            "name": codepart.attributes["file_path"],
            "description": codepart.attributes["file_path"],
            "content": codepart.attributes["source_code"],
        }
        for codepart in retrieved_codeparts
    ]

## Cognee Search Module

This module contains the search function that is used to search for nodes in the graph. It supports various search types and integrates with user permissions to filter results accordingly.

### Search Types

The `SearchType` enum defines the different types of searches that can be performed:

- `ADJACENT`: Search for nodes adjacent to a given node.
- `TRAVERSE`: Traverse the graph to find related nodes.
- `SIMILARITY`: Find nodes similar to a given node.
- `SUMMARY`: Retrieve a summary of the node.
- `SUMMARY_CLASSIFICATION`: Classify the summary of the node.
- `NODE_CLASSIFICATION`: Classify the node.
- `DOCUMENT_CLASSIFICATION`: Classify the document.
- `CYPHER`: Perform a Cypher query on the graph.

### Search Parameters

The `SearchParameters` class is a Pydantic model that validates and holds the search parameters:

```python
class SearchParameters(BaseModel):
    search_type: SearchType
    params: Dict[str, Any]

    @field_validator("search_type", mode="before")
    def convert_string_to_enum(cls, value):
        if isinstance(value, str):
            return SearchType.from_str(value)
        return value
```

### Search Function

The `search` function is the main entry point for performing a search. It handles user authentication, retrieves document IDs for the user, and filters the search results based on user permissions.

```python
async def search(search_type: str, params: Dict[str, Any], user: User = None) -> List:
    if user is None:
        user = await get_default_user()
  
    own_document_ids = await get_document_ids_for_user(user.id)
    search_params = SearchParameters(search_type=search_type, params=params)
    search_results = await specific_search([search_params])

    from uuid import UUID

    filtered_search_results = []

    for search_result in search_results:
        document_id = search_result["document_id"] if "document_id" in search_result else None
        document_id = UUID(document_id) if type(document_id) == str else document_id

        if document_id is None or document_id in own_document_ids:
            filtered_search_results.append(search_result)

    return filtered_search_results
```

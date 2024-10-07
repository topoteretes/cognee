## Cognee Search Module

This module contains the search function that is used to search for nodes in the graph. It supports various search types and integrates with user permissions to filter results accordingly.

### Search Types

The `SearchType` enum defines the different types of searches that can be performed:

- `INSIGHTS`: Search for insights from the knowledge graph.
- `SUMMARIES`: Search for summaries of the texts provided.
- `CHUNKS`: Search for the whole chunks of data.


### Search Function

The `search` function is the main entry point for performing a search. It handles user authentication, retrieves document IDs for the user, and filters the search results based on user permissions.

```python
from cognee import search, SearchType
await search(SearchType.INSIGHTS, "your_query")
```

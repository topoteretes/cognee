DEFAULT_TOOLS = [
    {
        "type": "function",
        "name": "search",
        "description": "Search for information within the knowledge graph",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "The query to search for in the knowledge graph",
                },
                "search_type": {
                    "type": "string",
                    "description": "Type of search to perform",
                    "enum": [
                        "CODE",
                        "GRAPH_COMPLETION",
                        "NATURAL_LANGUAGE",
                    ],
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                },
                "datasets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of dataset names to search within",
                },
            },
            "required": ["search_query"],
        },
    },
    {
        "type": "function",
        "name": "cognify",
        "description": "Convert text into a knowledge graph or process all added content",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text content to be converted into a knowledge graph",
                },
                "ontology_file_path": {
                    "type": "string",
                    "description": "Path to a custom ontology file",
                },
                "custom_prompt": {
                    "type": "string",
                    "description": "Custom prompt for entity extraction and graph generation. If provided, this prompt will be used instead of the default prompts.",
                },
            },
            "required": ["text"],
        },
    },
    # Commented as dangerous
    # {
    #     "type": "function",
    #     "name": "prune",
    #     "description": "Prune memory",
    # },
]

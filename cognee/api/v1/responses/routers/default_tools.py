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
                "code_query": {
                    "type": "object",
                    "description": "Structured arguments used when search_type is CODE",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "query_facts",
                                "explore",
                                "traverse",
                                "find_path",
                                "impact_analysis",
                            ],
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["forward", "reverse", "both"],
                        },
                        "name": {"type": "string"},
                        "names": {"type": "array", "items": {"type": "string"}},
                        "node_ids": {"type": "array", "items": {"type": "string"}},
                        "kind": {"type": "string"},
                        "kinds": {"type": "array", "items": {"type": "string"}},
                        "file": {"type": "string"},
                        "file_prefix": {"type": "string"},
                        "repo": {"type": "string"},
                        "property": {"type": "string"},
                        "property_value": {
                            "type": "string",
                            "description": "Exact property value rendered as text",
                        },
                        "relation_types": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "target": {"type": "string"},
                        "max_depth": {"type": "integer"},
                        "max_nodes": {"type": "integer"},
                    },
                    "additionalProperties": True,
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
                "graph_model_name": {
                    "type": "string",
                    "description": "Name of the graph model to use",
                },
                "graph_model_file": {
                    "type": "string",
                    "description": "Path to a custom graph model file",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "prune",
        "description": "Remove unnecessary or outdated information from the knowledge graph",
        "parameters": {
            "type": "object",
            "properties": {
                "prune_strategy": {
                    "type": "string",
                    "enum": ["light", "moderate", "aggressive"],
                    "description": "Strategy for pruning the knowledge graph",
                    "default": "moderate",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence score to retain (0-1)",
                    "minimum": 0,
                    "maximum": 1,
                },
                "older_than": {
                    "type": "string",
                    "description": "ISO date string - prune nodes older than this date",
                },
            },
        },
    },
]

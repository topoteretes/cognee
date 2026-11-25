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
                        "HYBRID_COMPLETION",
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
                                "insights",
                                "architecture",
                                "delta",
                                "capabilities",
                                "search",
                                "neighbors",
                                "subgraph",
                                "flows",
                                "hubs",
                                "ask",
                            ],
                            "description": (
                                "ask answers a natural-language question with seeds, flows, "
                                "a one-hop neighbourhood and file:line citations; search ranks "
                                "symbols, routes and flows by name, docstring and path; "
                                "neighbors, subgraph, flows and hubs navigate from there; "
                                "capabilities reports what the graph and server support"
                            ),
                        },
                        "diagram": {
                            "type": "string",
                            "enum": ["mermaid", "dot"],
                            "description": (
                                "Also render the result as diagram source (returned under "
                                "'diagram'); architecture draws Mermaid by default"
                            ),
                        },
                        "source": {
                            "type": "string",
                            "description": "insights: explainer name (cycles, hotspots, ...)",
                        },
                        "min_confidence": {
                            "type": "number",
                            "description": "insights: keep findings scored at or above this (0-1)",
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
                        "text": {
                            "type": "string",
                            "description": "search: the free text to rank code by",
                        },
                        "question": {
                            "type": "string",
                            "description": "ask: the natural-language question about the code",
                        },
                        "seeds": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "subgraph: seed names (at most 20)",
                        },
                        "seed_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "subgraph: seed node ids or enola fact ids",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "subgraph: hops from the seeds (default 1, max 3)",
                        },
                        "max_edges": {
                            "type": "integer",
                            "description": "subgraph: edge limit (default 300, max 1000)",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["list", "show", "through"],
                            "description": (
                                "flows: list the ranked flows, show one flow with its stages, "
                                "or list the flows through a symbol"
                            ),
                        },
                        "symbol": {
                            "type": "string",
                            "description": "flows (mode through): the symbol the flows reach",
                        },
                        "include_tests": {
                            "type": "boolean",
                            "description": "search, ask, hubs: include test code (default false)",
                        },
                        "include_side": {
                            "type": "boolean",
                            "description": (
                                "flows: include side flows such as scripts and tooling "
                                "(default false)"
                            ),
                        },
                        "max_seeds": {
                            "type": "integer",
                            "description": "ask: search hits to start from (default 5, max 10)",
                        },
                        "max_citations": {
                            "type": "integer",
                            "description": "ask: citations to return (default 25, max 60)",
                        },
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

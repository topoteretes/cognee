import asyncio
import json
import os
import cognee
from cognee.modules.migration.sources.zep import GraphitiSource

async def main():
    # STEP 0: Reproducibility
    print("STEP 0: Pruning system for reproducibility...")
    await cognee.prune.prune_system(metadata=True)
    
    # STEP 1: Create a minimal but realistic sample Graphiti dump
    print("\nSTEP 1: Creating sample Graphiti dump with bi-temporal facts...")
    sample_dump = {
        "episodes": [
            {
                "uuid": "ep-1",
                "name": "Meeting 1",
                "content": "Alice stated that Bob is the new CEO.",
                "created_at": "2023-01-01T10:00:00Z"
            },
            {
                "uuid": "ep-2",
                "name": "Meeting 2",
                "content": "Alice stated that Charlie replaced Bob as CEO.",
                "created_at": "2024-01-01T10:00:00Z"
            }
        ],
        "entities": [
            {"uuid": "ent-1", "name": "Alice", "labels": ["Person"]},
            {"uuid": "ent-2", "name": "Bob", "labels": ["Person"]},
            {"uuid": "ent-3", "name": "Charlie", "labels": ["Person"]}
        ],
        "facts": [
            {
                "uuid": "fact-1",
                "source_node_uuid": "ent-1",
                "target_node_uuid": "ent-2",
                "name": "knows",
                "fact": "Alice knows Bob",
                "valid_at": "2023-01-01T10:00:00Z",
                "invalid_at": None
            },
            {
                "uuid": "fact-2",
                "source_node_uuid": "ent-2",
                "target_node_uuid": "ent-3",
                "name": "knows",
                "fact": "Bob knows Charlie",
                "valid_at": "2023-01-01T10:00:00Z",
                "invalid_at": None
            },
            {
                "uuid": "fact-3",
                "source_node_uuid": "ent-1",
                "target_node_uuid": "ent-2",
                "name": "reports to",
                "fact": "Alice reports to Bob",
                "valid_at": "2023-01-01T10:00:00Z",
                "invalid_at": "2024-01-01T10:00:00Z",
                "episodes": ["ep-1"]
            },
            {
                "uuid": "fact-4",
                "source_node_uuid": "ent-1",
                "target_node_uuid": "ent-3",
                "name": "reports to",
                "fact": "Alice reports to Charlie",
                "valid_at": "2024-01-01T10:00:00Z",
                "invalid_at": None,
                "episodes": ["ep-2"]
            }
        ]
    }
    
    # STEP 2: Load it using GraphitiSource
    print("\nSTEP 2: Loading Graphiti dump into Cognee (mode='preserve')...")
    source = GraphitiSource(sample_dump, mode="preserve")
    await cognee.remember(source)
    
    # STEP 3: Call cognee.cognify() to build the graph
    print("\nSTEP 3: Cognifying imported data to build the graph...")
    await cognee.cognify()
    
    # STEP 4: Recall to verify migrated content
    print("\nSTEP 4: Verifying migrated content via recall()...")
    
    print("\n--- Recalling Entity (Alice) ---")
    entity_results = await cognee.recall("Who is Alice?")
    for res in entity_results:
        print(res)
    
    print("\n--- Recalling Fact/Relation (Who does Alice report to?) ---")
    fact_results = await cognee.recall("Who does Alice report to?")
    for res in fact_results:
        print(res)
    
    print("\n--- Recalling Episode Content (What happened in Meeting 1?) ---")
    episode_results = await cognee.recall("What happened in Meeting 1?")
    for res in episode_results:
        print(res)
    
    # STEP 5 & 6: Explanation block
    print("\n" + "="*60)
    print("BI-TEMPORAL MAPPING & LIMITATIONS")
    print("============================================================")
    print("Bi-Temporal Mapping:")
    print("Graphiti uses 'valid_at' and 'invalid_at' timestamps for facts.")
    print("Notice that fact-3 (Alice reports to Bob) has an invalid_at")
    print("timestamp, marking it as a superseded/expired fact. These")
    print("fields map directly to COGXFact's temporal metadata in Cognee.")
    print("\nCurrent Limitations:")
    print("This tutorial reads a FILE DUMP of a Graphiti graph, rather than")
    print("connecting to a live graphiti-core instance.")
    print("For live connections, see issue #3404.")
    print("\nImport Modes:")
    print("We used mode='preserve' which keeps the verbatim episodes.")
    print("Other options include 're-derive' (re-run LLM extraction) or")
    print("'hybrid' (keep episodes and graph).")

if __name__ == "__main__":
    asyncio.run(main())

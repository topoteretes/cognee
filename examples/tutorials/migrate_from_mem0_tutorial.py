import asyncio
import json
import os
import cognee
from cognee.modules.migration.sources.mem0 import Mem0Source

async def main():
    # STEP 0: Reproducibility
    print("STEP 0: Pruning system for reproducibility...")
    await cognee.prune.prune_system(metadata=True)
    
    # STEP 1: Create a minimal but realistic sample mem0 dump
    print("\nSTEP 1: Creating sample mem0 dump...")
    sample_dump = [
        {
            "id": "mem-1",
            "memory": "Alice lives in New York.",
            "categories": ["location", "personal"],
            "user_id": "alice_123",
            "metadata": {"source": "email", "confidence": 0.9},
            "created_at": "2023-10-01T12:00:00Z"
        },
        {
            "id": "mem-2",
            "memory": "Bob is a software engineer working at TechCorp.",
            "user_id": "bob_456",
            "metadata": {"source": "resume"},
            "created_at": "2023-10-02T09:30:00Z"
        },
        {
            "id": "mem-3",
            "memory": "Alice and Bob had a meeting about the new AI project on Tuesday.",
            "user_id": "alice_123",
            "metadata": {"type": "episodic", "participants": ["alice_123", "bob_456"]},
            "created_at": "2023-10-03T14:15:00Z"
        },
        {
            "id": "mem-4",
            "memory": "The new AI project requires a background in graph databases.",
            "user_id": "bob_456",
            "metadata": {"type": "factual", "project": "AI_Project"},
            "created_at": "2023-10-04T10:00:00Z"
        },
        {
            "id": "mem-5",
            "memory": "Alice prefers to drink oat milk lattes.",
            "user_id": "alice_123",
            "metadata": {"source": "chat", "category": "preferences"},
            "created_at": "2023-10-05T08:45:00Z"
        }
    ]
    
    # STEP 2: Show all THREE modes explicitly
    print("\nSTEP 2: Demonstrating import modes for Mem0")
    print("Mode 1: 'preserve' - maps memories straight to graph nodes, no LLM re-processing.")
    print("Loading mem0 dump into Cognee using mode='preserve'...")
    source = Mem0Source(sample_dump, mode="preserve")
    await cognee.remember(source)
    
    # Mode 2: re-derive - runs cognify on raw text, rebuilds graph
    # await cognee.remember(Mem0Source(sample_dump, mode="re-derive"))
    
    # Mode 3: hybrid - preserve structure + re-derive enrichment
    # await cognee.remember(Mem0Source(sample_dump, mode="hybrid"))
    
    # STEP 3: Call cognee.cognify() to build the graph
    print("\nSTEP 3: Cognifying imported data to build the graph...")
    await cognee.cognify()
    
    # STEP 4: Recall to verify migrated content
    print("\nSTEP 4: Verifying migrated content via recall()...")
    
    # Query by a specific memory's content
    print("\n--- Recalling Specific Memory Content (What does Alice like to drink?) ---")
    drink_results = await cognee.recall("What does Alice like to drink?")
    for res in drink_results:
        print(res)
    
    # Query by user context
    print("\n--- Recalling User Context (What do we know about Bob?) ---")
    bob_results = await cognee.recall("What do we know about Bob?")
    for res in bob_results:
        print(res)
    
    # Query something that requires connecting two memories
    print("\n--- Recalling Connected Memories (Who is working on the AI project?) ---")
    ai_results = await cognee.recall("Who is working on the AI project?")
    for res in ai_results:
        print(res)
        
    # STEP 5 & 6: Print all results with clear labels and Explanation block
    print("\n" + "="*60)
    print("COGXMEMORY MAPPING, IMPORT MODES & LIMITATIONS")
    print("============================================================")
    print("COGXMemory Mapping:")
    print("Mem0 memories land specifically as 'COGXMemory' records in Cognee,")
    print("rather than generic 'COGXFact' or 'COGXEntity'. This allows them")
    print("to preserve contextual details like user_id and categories.")
    print("\nImport Modes:")
    print("1. 'preserve': Skips LLM extraction and imports memories directly")
    print("   as graph nodes. Fast and maintains exactly what Mem0 exported.")
    print("2. 're-derive': Treats memories as raw text and runs the full LLM")
    print("   extraction pipeline to rebuild a new graph from scratch.")
    print("3. 'hybrid': Keeps the original Mem0 structure while also using the")
    print("   LLM to enrich the graph with additional connections.")
    print("\nCurrent Limitations:")
    print("This tutorial reads a static list (file dump) of a Mem0 export.")
    print("For live Mem0 API integration, see issue #3403.")

if __name__ == "__main__":
    asyncio.run(main())

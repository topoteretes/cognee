import asyncio
from cognee.api.v1.remember import remember
from cognee.api.v1.search import search
from cognee.api.v1.prune import prune_data
from cognee.modules.migration.sources.mem0 import Mem0Source

async def main():
    print("🧹 Cleaning up old data...")
    await prune_data()

    print("\n📦 Loading Mem0 export...")
    # This is a sample Mem0 export format.
    mem0_dump = {
        "results": [
            {
                "id": "m1",
                "memory": "User prefers dark mode and uses VS Code.",
                "user_id": "u123",
                "categories": ["preferences"],
                "created_at": "2024-05-01T12:00:00Z"
            },
            {
                "id": "m2",
                "memory": "User is allergic to peanuts.",
                "user_id": "u123",
                "categories": ["health"],
                "created_at": "2024-05-02T15:30:00Z"
            }
        ]
    }

    # Step 1: Import Mem0 export into Cognee.
    # mode="preserve" maps memories directly to graph nodes (COGXMemory) without LLM extraction.
    print("📥 Importing Mem0 memories into Cognee...")
    source = Mem0Source(mem0_dump, mode="preserve")
    await remember(source)

    # Step 2: Query the imported memories
    print("\n🔍 Querying migrated memories...")
    
    # We can perform a semantic search to recall memories
    query_text = "What is the user allergic to?"
    print(f"\nQuery: {query_text}")
    results = await search("SIMILARITY", query_text=query_text)
    
    if results:
        for result in results:
            print(f"- Found memory: {result}")
    else:
        print("No results found. (Make sure your embeddings are configured properly)")
        
    print("\n✅ Mem0 Migration Tutorial Complete!")

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from cognee.api.v1.remember import remember
from cognee.api.v1.search import search
from cognee.api.v1.prune import prune_data
from cognee.modules.migration.sources.zep import GraphitiSource

async def main():
    print("🧹 Cleaning up old data...")
    await prune_data()

    print("\n📦 Loading Graphiti export...")
    # This is a sample Graphiti JSON export containing entities, facts, and episodes.
    graphiti_dump = {
        "episodes": [
            {
                "uuid": "ep-1",
                "name": "Meeting 1",
                "content": "Alice told Bob about the project plan.",
                "created_at": "2024-05-01T10:00:00Z",
                "user_id": "user1"
            }
        ],
        "entities": [
            {
                "uuid": "ent-1",
                "name": "Alice",
                "labels": ["Person", "Entity"],
                "summary": "Project manager",
                "created_at": "2024-05-01T10:01:00Z"
            },
            {
                "uuid": "ent-2",
                "name": "Bob",
                "labels": ["Person", "Entity"],
                "summary": "Software engineer",
                "created_at": "2024-05-01T10:01:00Z"
            }
        ],
        "facts": [
            {
                "uuid": "fact-1",
                "source_node_uuid": "ent-1",
                "target_node_uuid": "ent-2",
                "relation": "communicated_with",
                "fact": "Alice told Bob about the project plan.",
                "valid_at": "2024-05-01T10:00:00Z",
                "created_at": "2024-05-01T10:02:00Z",
                "episodes": ["ep-1"]
            }
        ]
    }

    # Step 1: Import Graphiti export into Cognee.
    # mode="preserve" or "hybrid" can be used. GraphitiSource defaults to "hybrid".
    print("📥 Importing Graphiti graph into Cognee...")
    source = GraphitiSource(graphiti_dump, mode="preserve")
    await remember(source)

    # Step 2: Query the imported facts/entities
    print("\n🔍 Querying migrated knowledge graph...")
    
    # We can perform a semantic search to recall graph nodes
    query_text = "Who did Alice communicate with?"
    print(f"\nQuery: {query_text}")
    results = await search("SIMILARITY", query_text=query_text)
    
    if results:
        for result in results:
            print(f"- Found graph item: {result}")
    else:
        print("No results found. (Make sure your embeddings are configured properly)")
        
    print("\n✅ Graphiti Migration Tutorial Complete!")

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from cognee.api.v1.remember import remember
from cognee.api.v1.search import search
from cognee.api.v1.prune import prune_data
from cognee.modules.migration.sources.letta import LettaSource
from cognee.modules.migration.sources.zep import ZepSource

async def main():
    print("🧹 Cleaning up old data...")
    await prune_data()

    print("\n📦 Loading Letta/MemGPT export...")
    letta_dump = {
        "agents": [
            {
                "name": "Assistant",
                "core_memory": [
                    {
                        "label": "persona",
                        "value": "I am a friendly assistant."
                    },
                    {
                        "label": "human",
                        "value": "User is a data scientist from New York."
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "Can you remind me where I'm from?"
                    },
                    {
                        "role": "assistant",
                        "content": "You are from New York!"
                    }
                ]
            }
        ]
    }

    print("📥 Importing Letta agent into Cognee...")
    # Letta mode defaults to "re-derive", which processes documents and memory blocks.
    letta_source = LettaSource(letta_dump, mode="preserve")
    await remember(letta_source)

    print("\n📦 Loading Zep export...")
    zep_dump = {
        "episodes": [
            {
                "uuid": "ep-1",
                "name": "Chat Session",
                "content": "User discussed their recent trip to Japan.",
                "created_at": "2024-05-01T10:00:00Z"
            }
        ],
        "entities": [
            {
                "uuid": "ent-1",
                "name": "Japan",
                "labels": ["Location", "Entity"],
                "summary": "A country in East Asia."
            }
        ],
        "facts": []
    }

    print("📥 Importing Zep memories into Cognee...")
    # Zep mode defaults to "hybrid". We can use "preserve" for direct graph loading.
    zep_source = ZepSource(zep_dump, mode="preserve")
    await remember(zep_source)

    # Query the imported memories
    print("\n🔍 Querying migrated knowledge graph...")
    
    query_text = "Where is the user from?"
    print(f"\nQuery: {query_text}")
    results = await search("SIMILARITY", query_text=query_text)
    
    if results:
        for result in results:
            print(f"- Found: {result}")
    else:
        print("No results found. (Make sure your embeddings are configured properly)")
        
    print("\n✅ Letta & Zep Migration Tutorial Complete!")

if __name__ == "__main__":
    asyncio.run(main())

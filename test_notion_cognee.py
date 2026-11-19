import asyncio
import os
import cognee
from notion_loader import NotionLoader

async def main():
    # Asigură-te că ai setat cheia: export NOTION_API_KEY="secret_xxx"
    notion_loader = NotionLoader()
    
    print("Se preiau datele din Notion...")
    pages = notion_loader.load_data()
    print(S-au găsit {len(pages)} pagini.)

    for page in pages:
        doc_text = f"Titlu: {page['title']}\n\n{page['content']}"
        # Adăugăm datele în Cognee
        await cognee.add(doc_text)

    print("Se procesează datele în Knowledge Graph (cognify)...")
    await cognee.cognify()

    print("Testăm căutarea în memorie...")
    results = await cognee.search("Ce informații am în paginile din Notion?")
    print("Rezultate:", results)

if __name__ == "__main__":
    asyncio.run(main())
import os
import litellm
import httpx
client = httpx.AsyncClient(
    headers={
        "Authorization": f"Bearer {os.getenv('LLM_API_KEY')}",
        "api-key": os.getenv('LLM_API_KEY')
    },
    verify=os.getenv('REQUESTS_CA_BUNDLE'),  # opcional
)

litellm.aclient_session = client
import cognee
import asyncio

async def main():
    # Add text to cognee
    await cognee.add("Natural language processing (NLP) is an interdisciplinary subfield of computer science and information retrieval.")

    # Generate the knowledge graph
    await cognee.cognify()

    # Query the knowledge graph
    results = await cognee.search("Tell me about NLP")

    # Display the results
    for result in results:
        print(result)


if __name__ == '__main__':
    asyncio.run(main())
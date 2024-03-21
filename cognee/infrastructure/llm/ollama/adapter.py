import aiohttp
from typing import List, Type
from pydantic import BaseModel
from cognee.infrastructure.llm.llm_interface import LLMInterface
class OllamaAPIConnector(LLMInterface):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}'}

    async def async_get_embedding_with_backoff(self, text, model="text-embedding-ada-002"):
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/embeddings"
            payload = {"model": model, "prompt": text}
            async with session.post(url, json=payload, headers=self.headers) as response:
                return await response.json()

    def get_embedding_with_backoff(self, text: str, model: str = "text-embedding-ada-002"):
        # Synchronous version not implemented for this async example
        raise NotImplementedError

    async def async_get_batch_embeddings_with_backoff(self, texts: List[str], models: List[str]):
        # This is a simplified version. In practice, you should handle different models and batching.
        results = []
        async with aiohttp.ClientSession() as session:
            for text, model in zip(texts, models):
                url = f"{self.base_url}/api/embeddings"
                payload = {"model": model, "prompt": text}
                async with session.post(url, json=payload, headers=self.headers) as response:
                    results.append(await response.json())
        return results

    async def acreate_structured_output(self, text_input: str, system_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        async with aiohttp.ClientSession() as session:
            # Assuming the API has an endpoint for structured responses, adjust accordingly.
            url = f"{self.base_url}/api/structured_output"
            payload = {"prompt": text_input, "system": system_prompt}
            async with session.post(url, json=payload, headers=self.headers) as response:
                response_data = await response.json()
                return response_model(**response_data)

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        # This method is typically synchronous and does not need to be async
        return f"{system_prompt}\nUser: {text_input}\nAssistant:"

# Example usage
# connector = OllamaAPIConnector(base_url="http://your_ollama_base_url", api_key="your_api_key")
# response = await connector.async_get_embedding_with_backoff(text="Why is the sky blue?")
# print(response)

"""Adapter for OpenAI's GPT-3, GPT=4 API."""
import asyncio
from typing import List, Type
import openai
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt
from cognitive_architecture.utils import read_query_prompt
from ..llm_interface import LLMInterface

class OpenAIAdapter(LLMInterface):
    """Adapter for OpenAI's GPT-3, GPT=4 API"""
    def __init__(self, api_key: str, model:str):
        openai.api_key = api_key
        self.aclient = instructor.apatch(AsyncOpenAI())
        self.model = model

    @retry(stop = stop_after_attempt(5))
    def completions_with_backoff(self, **kwargs):
        """Wrapper around ChatCompletion.create w/ backoff"""
        # Local model
        return openai.chat.completions.create(**kwargs)

    @retry(stop = stop_after_attempt(5))
    async def acompletions_with_backoff(self,**kwargs):
        """Wrapper around ChatCompletion.acreate w/ backoff"""
        return await openai.chat.completions.acreate(**kwargs)

    @retry(stop = stop_after_attempt(5))
    async def acreate_embedding_with_backoff(self, input: List[str], model: str = "text-embedding-ada-002"):
        """Wrapper around Embedding.acreate w/ backoff"""

        return await self.aclient.embeddings.create(input=input, model=model)

    async def async_get_embedding_with_backoff(self, text, model="text-embedding-ada-002"):
        """To get text embeddings, import/call this function
        It specifies defaults + handles rate-limiting + is async"""
        text = text.replace("\n", " ")
        response = await self.aclient.embeddings.create(input =text, model= model)
        embedding = response.data[0].embedding
        return embedding

    @retry(stop = stop_after_attempt(5))
    def create_embedding_with_backoff(self, **kwargs):
        """Wrapper around Embedding.create w/ backoff"""
        return openai.embeddings.create(**kwargs)

    def get_embedding_with_backoff(self, text: str, model: str = "text-embedding-ada-002"):
        """To get text embeddings, import/call this function
        It specifies defaults + handles rate-limiting
        :param text: str
        :param model: str
        """
        text = text.replace("\n", " ")
        response = self.create_embedding_with_backoff(input=[text], model=model)
        embedding = response.data[0].embedding
        return embedding

    async def async_get_batch_embeddings_with_backoff(self, texts: List[str], models: List[str]):
        """To get multiple text embeddings in parallel, import/call this function
        It specifies defaults + handles rate-limiting + is async"""
        # Collect all coroutines
        coroutines = (self.async_get_embedding_with_backoff(text, model)
            for text, model in zip(texts, models))

        # Run the coroutines in parallel and gather the results
        embeddings = await asyncio.gather(*coroutines)

        return embeddings

    @retry(stop = stop_after_attempt(5))
    async def acreate_structured_output(self, text_input: str, system_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        """Generate a response from a user query."""
        return await self.aclient.chat.completions.create(
            model = self.model,
            messages = [
                {
                    "role": "user",
                    "content": f"""Use the given format to
                    extract information from the following input: {text_input}. """,
                },
                {"role": "system", "content": system_prompt},
            ],
            response_model = response_model,
        )

    def show_prompt(self, text_input: str, system_prompt_path: str) -> str:
        """Format and display the prompt for a user query."""
        if not text_input:
            text_input= "No user input provided."
        if not system_prompt_path:
            raise ValueError("No system prompt path provided.")
        system_prompt = read_query_prompt(system_prompt_path)

        formatted_prompt = f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
        return formatted_prompt

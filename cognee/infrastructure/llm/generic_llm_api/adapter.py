'''Adapter for Generic API LLM provider API'''
import asyncio
from typing import List, Type
from pydantic import BaseModel
import instructor
from tenacity import retry, stop_after_attempt
import openai
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.shared.data_models import MonitoringTool
from cognee.base_config import get_base_config
from cognee.infrastructure.llm.config import get_llm_config


class GenericAPIAdapter(LLMInterface):
    """Adapter for Generic API LLM provider API """
    name: str
    model: str
    api_key: str

    def __init__(self, api_endpoint, api_key: str, model: str, name: str):
        self.name = name
        self.model = model
        self.api_key = api_key

        llm_config = get_llm_config()

        if llm_config.llm_provider == "groq":
            from groq import groq
            self.aclient = instructor.from_openai(
                client = groq.Groq(
                  api_key = api_key,
                ),
                mode = instructor.Mode.MD_JSON
            )
        else:
            base_config = get_base_config()

            if base_config.monitoring_tool == MonitoringTool.LANGFUSE:
                from langfuse.openai import AsyncOpenAI
            elif base_config.monitoring_tool == MonitoringTool.LANGSMITH:
                from langsmith import wrappers
                from openai import AsyncOpenAI
                AsyncOpenAI = wrappers.wrap_openai(AsyncOpenAI())
            else:
                from openai import AsyncOpenAI

            self.aclient = instructor.patch(
                AsyncOpenAI(
                    base_url = api_endpoint,
                    api_key = api_key,  # required, but unused
                ),
                mode = instructor.Mode.JSON,
            )

    @retry(stop = stop_after_attempt(5))
    def completions_with_backoff(self, **kwargs):
        """Wrapper around ChatCompletion.create w/ backoff"""
        # Local model
        return openai.chat.completions.create(**kwargs)

    @retry(stop = stop_after_attempt(5))
    async def acompletions_with_backoff(self, **kwargs):
        """Wrapper around ChatCompletion.acreate w/ backoff"""
        return await openai.chat.completions.acreate(**kwargs)

    @retry(stop = stop_after_attempt(5))
    async def acreate_embedding_with_backoff(self, input: List[str], model: str = "text-embedding-3-large"):
        """Wrapper around Embedding.acreate w/ backoff"""

        return await self.aclient.embeddings.create(input = input, model = model)

    async def async_get_embedding_with_backoff(self, text, model="text-embedding-3-large"):
        """To get text embeddings, import/call this function
        It specifies defaults + handles rate-limiting + is async"""
        text = text.replace("\n", " ")
        response = await self.aclient.embeddings.create(input = text, model = model)
        embedding = response.data[0].embedding
        return embedding

    @retry(stop = stop_after_attempt(5))
    def create_embedding_with_backoff(self, **kwargs):
        """Wrapper around Embedding.create w/ backoff"""
        return openai.embeddings.create(**kwargs)

    def get_embedding_with_backoff(self, text: str, model: str = "text-embedding-3-large"):
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

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""
        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise ValueError("No system prompt path provided.")
        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n""" if system_prompt else None
        return formatted_prompt

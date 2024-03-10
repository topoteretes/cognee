"""Adapter for OpenAI's GPT-3, GPT=4 API."""
import os
import time
import random
import asyncio
from typing import List, Type
import openai
import instructor
from openai import OpenAI,AsyncOpenAI
from pydantic import BaseModel
from cognitive_architecture.config import Config
from cognitive_architecture.utils import read_query_prompt
from ..llm_interface import LLMInterface

#
# config = Config()
# config.load()

# aclient = instructor.apatch(AsyncOpenAI())
# OPENAI_API_KEY = config.openai_key

class OpenAIAdapter(LLMInterface):
    """Adapter for OpenAI's GPT-3, GPT=4 API"""
    def __init__(self, api_key: str, model:str):
        openai.api_key = api_key
        self.aclient = instructor.apatch(AsyncOpenAI())
        self.model = model
        # OPENAI_API_KEY = config.openai_key

    @staticmethod
    def retry_with_exponential_backoff(
            func,
            initial_delay: float = 1,
            exponential_base: float = 2,
            jitter: bool = True,
            max_retries: int = 20,
            errors: tuple = (openai.RateLimitError,),
    ):
        """Retry a function with exponential backoff."""

        def wrapper(*args, **kwargs):
            """Wrapper for sync functions."""
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper


    @staticmethod
    async def aretry_with_exponential_backoff(
            func,
            initial_delay: float = 1,
            exponential_base: float = 2,
            jitter: bool = True,
            max_retries: int = 20,
            errors: tuple = (openai.RateLimitError,),
    ):
        """Retry a function with exponential backoff."""

        async def wrapper(*args, **kwargs):
            """Wrapper for async functions.
            :param args: list
            :param kwargs: dict"""
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return await func(*args, **kwargs)

                # Retry on specified errors
                except errors as e:
                    print(f"acreate (backoff): caught error: {e}")
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    await asyncio.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper


    @retry_with_exponential_backoff
    def completions_with_backoff(self, **kwargs):
        """Wrapper around ChatCompletion.create w/ backoff"""
        # Local model
        return openai.chat.completions.create(**kwargs)

    @aretry_with_exponential_backoff
    async def acompletions_with_backoff(self,**kwargs):
        """Wrapper around ChatCompletion.acreate w/ backoff"""
        return await openai.chat.completions.acreate(**kwargs)

    @aretry_with_exponential_backoff
    async def acreate_embedding_with_backoff(self, input: List[str], model: str = "text-embedding-ada-002"):
        """Wrapper around Embedding.acreate w/ backoff"""

        # client = openai.AsyncOpenAI(
        #     # This is the default and can be omitted
        #     api_key=os.environ.get("OPENAI_API_KEY"),
        # )

        return await self.aclient.embeddings.create(input=input, model=model)

    async def async_get_embedding_with_backoff(self, text, model="text-embedding-ada-002"):
        """To get text embeddings, import/call this function
        It specifies defaults + handles rate-limiting + is async"""
        text = text.replace("\n", " ")
        print(text)
        print(model)
        response = await self.aclient.embeddings.create(input =text, model= model)
        # response = await self.acreate_embedding_with_backoff(input=text, model=model)
        embedding = response.data[0].embedding
        return embedding

    @retry_with_exponential_backoff
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
        # Create a generator of coroutines
        coroutines = (await self.async_get_embedding_with_backoff(text, model)
                      for text, model in zip(texts, models))

        # Run the coroutines in parallel and gather the results
        embeddings = await asyncio.gather(*coroutines)

        return embeddings

    async def acreate_structured_output(self, text_input: str, system_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        """Generate a response from a user query."""


        return await self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Use the given format to
                     extract information from the following input: {text_input}. """,
                },
                {"role": "system", "content": system_prompt},
            ],
            response_model=response_model,
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

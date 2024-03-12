"""Tools for interacting with OpenAI's GPT-3, GPT-4 API"""
import asyncio
import os
from typing import List
from tenacity import retry, stop_after_attempt
import openai

HOST = os.getenv("OPENAI_API_BASE")
HOST_TYPE = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion

if HOST is not None:
    openai.api_base = HOST

@retry(stop = stop_after_attempt(5))
def completions_with_backoff(**kwargs):
    """Wrapper around ChatCompletion.create w/ backoff"""
    # Local model
    return openai.chat.completions.create(**kwargs)

@retry(stop = stop_after_attempt(5))
async def acompletions_with_backoff(**kwargs):
    """Wrapper around ChatCompletion.acreate w/ backoff"""
    return await openai.chat.completions.acreate(**kwargs)


@retry(stop = stop_after_attempt(5))
async def acreate_embedding_with_backoff(**kwargs):
    """Wrapper around Embedding.acreate w/ backoff"""

    client = openai.AsyncOpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    return await client.embeddings.create(**kwargs)


async def async_get_embedding_with_backoff(text, model="text-embedding-ada-002"):
    """To get text embeddings, import/call this function
    It specifies defaults + handles rate-limiting + is async"""
    text = text.replace("\n", " ")
    response = await acreate_embedding_with_backoff(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding


@retry(stop = stop_after_attempt(5))
def create_embedding_with_backoff(**kwargs):
    """Wrapper around Embedding.create w/ backoff"""
    return openai.embeddings.create(**kwargs)


def get_embedding_with_backoff(text:str, model:str="text-embedding-ada-002"):
    """To get text embeddings, import/call this function
    It specifies defaults + handles rate-limiting
    :param text: str
    :param model: str
    """
    text = text.replace("\n", " ")
    response = create_embedding_with_backoff(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding



async def async_get_batch_embeddings_with_backoff(texts: List[str], models: List[str]) :
    """To get multiple text embeddings in parallel, import/call this function
    It specifies defaults + handles rate-limiting + is async"""
    # Create a generator of coroutines
    coroutines = (async_get_embedding_with_backoff(text, model) for text, model in zip(texts, models))

    # Run the coroutines in parallel and gather the results
    embeddings = await asyncio.gather(*coroutines)

    return embeddings

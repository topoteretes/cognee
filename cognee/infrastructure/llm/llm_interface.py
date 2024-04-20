""" LLM Interface """

from typing import Type, Protocol
from abc import abstractmethod
from pydantic import BaseModel
class LLMInterface(Protocol):
    """ LLM Interface """

    # @abstractmethod
    # async def async_get_embedding_with_backoff(self, text, model="text-embedding-ada-002"):
    #     """To get text embeddings, import/call this function"""
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def get_embedding_with_backoff(self, text: str, model: str = "text-embedding-ada-002"):
    #     """To get text embeddings, import/call this function"""
    #     raise NotImplementedError
    #
    # @abstractmethod
    # async def async_get_batch_embeddings_with_backoff(self, texts: List[str], models: List[str]):
    #     """To get multiple text embeddings in parallel, import/call this function"""
    #     raise NotImplementedError

    # """ Get completions """
    # async def acompletions_with_backoff(self, **kwargs):
    #     raise NotImplementedError
    #
    """ Structured output """
    @abstractmethod
    async def acreate_structured_output(self,
                                        text_input: str,
                                        system_prompt: str,
                                        response_model: Type[BaseModel]) -> BaseModel:
        """To get structured output, import/call this function"""
        raise NotImplementedError

    @abstractmethod
    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """To get structured output, import/call this function"""
        raise NotImplementedError

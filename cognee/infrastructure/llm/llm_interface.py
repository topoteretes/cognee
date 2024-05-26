""" LLM Interface """

from typing import Type, Protocol
from abc import abstractmethod
from pydantic import BaseModel
class LLMInterface(Protocol):
    """ LLM Interface """

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

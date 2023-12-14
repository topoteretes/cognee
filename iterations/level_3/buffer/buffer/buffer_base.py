import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv

load_dotenv()
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from typing import Optional, Dict, List, Union

import tracemalloc

tracemalloc.start()

import os

import uuid






class EpisodicBuffer(DynamicBaseMemory):
    def __init__(
        self,
        user_id: str,
        memory_id: Optional[str],
        index_name: Optional[str],
        db_type: str = "weaviate",
    ):
        super().__init__('EpisodicBuffer',
            user_id, memory_id, index_name, db_type, namespace="BUFFERMEMORY"
        )

        self.st_memory_id = str( uuid.uuid4())
        self.llm = ChatOpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4-0613",
            # callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
        )
        self.llm_base = OpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4-0613",
        )



    async def handle_modulator(
        self,
        modulator_name: str,
        attention_modulators: Dict[str, float],
        observation: str,
        namespace: Optional[str] = None,
        memory: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Union[str, float]]]:
        """
        Handle the given modulator based on the observation and namespace.

        Parameters:
        - modulator_name: Name of the modulator to handle.
        - attention_modulators: Dictionary of modulator values.
        - observation: The current observation.
        - namespace: An optional namespace.

        Returns:
        - Result of the modulator if criteria met, else None.
        """
        modulator_value = attention_modulators.get(modulator_name, 0.0)
        modulator_functions = {
            "freshness": lambda obs, ns, mem: self.freshness(observation=obs, namespace=ns, memory=mem),
            "frequency": lambda obs, ns, mem: self.frequency(observation=obs, namespace=ns, memory=mem),
            "relevance": lambda obs, ns, mem: self.relevance(observation=obs, namespace=ns, memory=mem),
            "saliency": lambda obs, ns, mem: self.saliency(observation=obs, namespace=ns, memory=mem),
        }

        result_func = modulator_functions.get(modulator_name)
        if not result_func:
            return None

        result = await result_func(observation, namespace, memory)
        if not result:
            return None

        try:
            logging.info("Modulator %s", modulator_name)
            logging.info("Modulator value %s", modulator_value)
            logging.info("Result %s", result[0])
            if  float(result[0]) >= float(modulator_value):
                return result
        except ValueError:
            pass

        return None

    async def available_operations(self) -> list[str]:
        """Determines what operations are available for the user to process PDFs"""

        return [
            "retrieve over time",
            "save to personal notes",
            "translate to german"
            # "load to semantic memory",
            # "load to episodic memory",
            # "load to buffer",
        ]

import numpy as np
# Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client
import json
from enum import Enum
from io import BytesIO
from typing import Dict, List, Union, Any
import logging

logging.basicConfig(level=logging.INFO)
import marvin
import requests
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.document_loaders import PyPDFLoader
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import WeaviateHybridSearchRetriever
from langchain.tools import tool
from marvin import ai_classifier
from pydantic import parse_obj_as
from weaviate.gql.get import HybridFusion
import numpy as np
load_dotenv()
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from typing import Optional, Dict, List, Union

import tracemalloc

tracemalloc.start()

import os
from datetime import datetime
from langchain import PromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.schema import Document, SystemMessage, HumanMessage
import uuid
import humanize
import weaviate

class DifferentiableLayer:
    def __init__(self, attention_modulators: dict):
        self.weights = {modulator: 1.0 for modulator in attention_modulators}
        self.learning_rate = 0.1
        self.regularization_lambda = 0.01
        self.weight_decay = 0.99

    async def adjust_weights(self, feedbacks: list[float]):
        """
        Adjusts the weights of the attention modulators based on user feedbacks.

        Parameters:
        - feedbacks: A list of feedback scores (between 0 and 1).
        """
        avg_feedback = np.mean(feedbacks)
        feedback_diff = 1.0 - avg_feedback

        # Adjust weights based on average feedback
        for modulator in self.weights:
            self.weights[modulator] += self.learning_rate * (-feedback_diff) - self.regularization_lambda * \
                                       self.weights[modulator]
            self.weights[modulator] *= self.weight_decay

        # Decaying the learning rate
        self.learning_rate *= 0.99

    async def get_weights(self):
        return self.weights



    async def _summarizer(self, text: str, document:str,  max_tokens: int = 1200):
        """Summarize text using OpenAI API, to reduce amount of code for modulators contributing to context"""
        class Summaries(BaseModel):
            """Schema for documentGroups"""
            summary: str = Field(
                ...,
                description="Summarized document")
        class SummaryContextList(BaseModel):
            """Buffer raw context processed by the buffer"""

            summaries: List[Summaries] = Field(..., description="List of summaries")
            observation: str = Field(..., description="The original user query")

        parser = PydanticOutputParser(pydantic_object=SummaryContextList)
        prompt = PromptTemplate(
            template=" \n{format_instructions}\nSummarize the observation briefly based on the user query, observation is: {query}\n. The document is: {document}",
            input_variables=["query", "document"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        _input = prompt.format_prompt(query=text, document=document)
        document_context_result = self.llm_base(_input.to_string())
        document_context_result_parsed = parser.parse(document_context_result)
        document_context_result_parsed = json.loads(document_context_result_parsed.json())
        document_summary = document_context_result_parsed["summaries"][0]["summary"]

        return document_summary

    async def memory_route(self, text_time_diff: str):
        @ai_classifier
        class MemoryRoute(Enum):
            """Represents classifer for freshness of memories"""

            data_uploaded_now = "1"
            data_uploaded_very_recently = "0.9"
            data_uploaded_recently = "0.7"
            data_uploaded_more_than_a_month_ago = "0.5"
            data_uploaded_more_than_three_months_ago = "0.3"
            data_uploaded_more_than_six_months_ago = "0.1"

        namespace = MemoryRoute(str(text_time_diff))

        return namespace

    async def freshness(self, observation: str, namespace: str = None, memory=None) -> list[str]:
        """Freshness - Score between 0 and 1  on how often was the information updated in episodic or semantic memory in the past"""
        logging.info("Starting with Freshness")

        lookup_value = await self.fetch_memories(
            observation=observation, namespace=namespace
        )
        unix_t = lookup_value["data"]["Get"]["EPISODICMEMORY"][0]["_additional"][
            "lastUpdateTimeUnix"
        ]

        # Convert Unix timestamp to datetime
        last_update_datetime = datetime.fromtimestamp(int(unix_t) / 1000)
        time_difference = datetime.now() - last_update_datetime
        time_difference_text = humanize.naturaltime(time_difference)
        namespace_ = await self.memory_route(str(time_difference_text))
        return [namespace_.value, lookup_value]

    async def frequency(self, observation: str, namespace: str, memory) -> list[str]:
        """Frequency - Score between 0 and 1 on how often was the information processed in episodic memory in the past
        Counts the number of times a memory was accessed in the past and divides it by the total number of memories in the episodic memory
        """
        logging.info("Starting with Frequency")
        weaviate_client = self.init_client(namespace=namespace)

        result_output = await self.fetch_memories(
            observation=observation, params=None, namespace=namespace
        )
        number_of_relevant_events = len(result_output["data"]["Get"]["EPISODICMEMORY"])
        number_of_total_events = (
            weaviate_client.query.aggregate(namespace).with_meta_count().do()
        )
        frequency = float(number_of_relevant_events) / float(
            number_of_total_events["data"]["Aggregate"]["EPISODICMEMORY"][0]["meta"][
                "count"
            ]
        )
        summary = await self._summarizer(text=observation, document=result_output["data"]["Get"]["EPISODICMEMORY"][0])
        logging.info("Frequency summary is %s", str(summary))
        return [str(frequency), summary]

    async def repetition(self, observation: str, namespace: str, memory) -> list[str]:
        """Repetition - Score between 0 and 1 based on how often and at what intervals a memory has been revisited.
        Accounts for the spacing effect, where memories accessed at increasing intervals are given higher scores.
        # TO DO -> add metadata column to make sure that the access is not equal to update, and run update vector function each time a memory is accessed
        """
        logging.info("Starting with Repetition")

        result_output = await self.fetch_memories(
            observation=observation, params=None, namespace=namespace
        )

        access_times = result_output["data"]["Get"]["EPISODICMEMORY"][0]["_additional"]["lastUpdateTimeUnix"]
        # Calculate repetition score based on access times
        if not access_times or len(access_times) == 1:
            return ["0", result_output["data"]["Get"]["EPISODICMEMORY"][0]]

        # Sort access times
        access_times = sorted(access_times)
        # Calculate intervals between consecutive accesses
        intervals = [access_times[i + 1] - access_times[i] for i in range(len(access_times) - 1)]
        # A simple scoring mechanism: Longer intervals get higher scores, as they indicate spaced repetition
        repetition_score = sum([1.0 / (interval + 1) for interval in intervals]) / len(intervals)
        summary = await self._summarizer(text = observation, document=result_output["data"]["Get"]["EPISODICMEMORY"][0])
        logging.info("Repetition is %s", str(repetition_score))
        logging.info("Repetition summary is %s", str(summary))
        return [str(repetition_score), summary]

    async def relevance(self, observation: str, namespace: str, memory) -> list[str]:
        """
        Fetches the fusion relevance score for a given observation from the episodic memory.
        Learn more about fusion scores here on Weaviate docs: https://weaviate.io/blog/hybrid-search-fusion-algorithms
        Parameters:
        - observation: The user's query or observation.
        - namespace: The namespace for the data.

        Returns:
        - The relevance score between 0 and 1.
        """
        logging.info("Starting with Relevance")
        score = memory["_additional"]["score"]
        logging.info("Relevance is %s", str(score))
        return [score, "fusion score"]

    async def saliency(self, observation: str, namespace=None, memory=None) -> list[str]:
        """Determines saliency by scoring the set of retrieved documents against each other and trying to determine saliency
        """
        logging.info("Starting with Saliency")
        class SaliencyRawList(BaseModel):
            """Schema for documentGroups"""
            summary: str = Field(
                ...,
                description="Summarized document")
            saliency_score: str = Field(
                None, description="The score between 0 and 1")
        class SailencyContextList(BaseModel):
            """Buffer raw context processed by the buffer"""

            docs: List[SaliencyRawList] = Field(..., description="List of docs")
            observation: str = Field(..., description="The original user query")

        parser = PydanticOutputParser(pydantic_object=SailencyContextList)
        prompt = PromptTemplate(
            template="Determine saliency of documents compared to the other documents retrieved \n{format_instructions}\nSummarize the observation briefly based on the user query, observation is: {query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        _input = prompt.format_prompt(query=observation)
        document_context_result = self.llm_base(_input.to_string())
        document_context_result_parsed = parser.parse(document_context_result)
        document_context_result_parsed = json.loads(document_context_result_parsed.json())
        saliency_score = document_context_result_parsed["docs"][0]["saliency_score"]
        saliency_values = document_context_result_parsed["docs"][0]["summary"]

        logging.info("Saliency is %s", str(saliency_score))
        logging.info("Saliency summary is %s", str(saliency_values))

        return [saliency_score, saliency_values]


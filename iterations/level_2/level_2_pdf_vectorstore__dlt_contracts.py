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

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")

# class MyCustomSyncHandler(BaseCallbackHandler):
#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")
#
#
# class MyCustomAsyncHandler(AsyncCallbackHandler):
#     """Async callback handler that can be used to handle callbacks from langchain."""
#
#     async def on_llm_start(
#             self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
#     ) -> None:
#         """Run when chain starts running."""
#         print("zzzz....")
#         await asyncio.sleep(0.3)
#         class_name = serialized["name"]
#         print("Hi! I just woke up. Your llm is starting")
#
#     async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
#         """Run when chain ends running."""
#         print("zzzz....")
#         await asyncio.sleep(0.3)
#         print("Hi! I just woke up. Your llm is ending")
#
#


# Assuming OpenAIEmbeddings and other necessary imports are available

from vectordb.basevectordb import  BaseMemory


from modulators.modulators import DifferentiableLayer


class SemanticMemory(BaseMemory):
    def __init__(
        self,
        user_id: str,
        memory_id: Optional[str],
        index_name: Optional[str],
        db_type: str = "weaviate",
    ):
        super().__init__(
            user_id, memory_id, index_name, db_type, namespace="SEMANTICMEMORY")


class EpisodicMemory(BaseMemory):
    def __init__(
        self,
        user_id: str,
        memory_id: Optional[str],
        index_name: Optional[str],
        db_type: str = "weaviate",
    ):
        super().__init__(
            user_id, memory_id, index_name, db_type, namespace="EPISODICMEMORY"
        )


class EpisodicBuffer(BaseMemory):
    def __init__(
        self,
        user_id: str,
        memory_id: Optional[str],
        index_name: Optional[str],
        db_type: str = "weaviate",
    ):
        super().__init__(
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




    # Example Usage
    # attention_modulators = {"freshness": 0.8, "frequency": 0.7, "relevance": 0.9, "saliency": 0.85}
    # diff_layer = DifferentiableLayer(attention_modulators)
    #
    # # Sample batch feedback
    # feedbacks = [0.75, 0.8, 0.9]
    #
    # # Adjust weights based on batch feedback
    # diff_layer.adjust_weights(feedbacks)
    #
    # print(diff_layer.get_weights())

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

    async def buffer_context(
        self,
        user_input=None,
        params=None,
        attention_modulators: dict = None,
    ):
        """Generates the context to be used for the buffer and passed to the agent"""
        # we just filter the data here to make sure input is clean
        prompt_filter = ChatPromptTemplate.from_template(
            """Filter and remove uneccessary information that is not relevant in the query to 
            the vector store to get more information, keep it as original as possbile: {query}"""
        )
        chain_filter = prompt_filter | self.llm
        output = await chain_filter.ainvoke({"query": user_input})

        # this part is partially done but the idea is to apply different attention modulators
        # to the data to fetch the most relevant information from the vector stores
        class BufferModulators(BaseModel):
            """Value of buffer modulators"""
            frequency: str = Field(..., description="Frequency score of the document")
            saliency: str = Field(..., description="Saliency score of the document")
            relevance: str = Field(..., description="Relevance score of the document")
            description:str = Field(..., description="Latest buffer modulators")
            direction: str= Field(..., description="Increase or a decrease of the modulator")

        parser = PydanticOutputParser(pydantic_object=BufferModulators)

        prompt = PromptTemplate(
            template="""Structure the buffer modulators to be used for the buffer. \n
                        {format_instructions} \nOriginal observation is: 
                        {query}\n """,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # check if modulators exist, initialize the modulators if needed
        if attention_modulators is None:
            # try:
            logging.info("Starting with attention mods")
            attention_modulators = await self.fetch_memories(observation="Attention modulators",
                                                         namespace="BUFFERMEMORY")

            logging.info("Attention modulators exist %s", str(attention_modulators))
            lookup_value_episodic = await self.fetch_memories(
                observation=str(output), namespace="EPISODICMEMORY"
            )
            # lookup_value_episodic= lookup_value_episodic["data"]["Get"]["EPISODICMEMORY"][0]["text"]
            prompt_classify = ChatPromptTemplate.from_template(
                """You are a classifier. Determine if based on the previous query if the user was satisfied with the output : {query}"""
            )
            json_structure = [{
                "name": "classifier",
                "description": "Classification indicating if it's output is satisfactory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "boolean",
                            "description": "The classification true or false"
                        }
                    }, "required": ["classification"]}
            }]
            chain_filter = prompt_classify | self.llm.bind(function_call= {"name": "classifier"}, functions= json_structure)
            classifier_output = await chain_filter.ainvoke({"query": lookup_value_episodic})
            arguments_str = classifier_output.additional_kwargs['function_call']['arguments']
            print("This is the arguments string", arguments_str)
            arguments_dict = json.loads(arguments_str)
            classfier_value = arguments_dict.get('classification', None)

            print("This is the classifier value", classfier_value)

            if classfier_value:
                # adjust the weights of the modulators by adding a positive value
                print("Lookup value, episodic", lookup_value_episodic["data"]["Get"]["EPISODICMEMORY"][0]["text"])
                prompt_classify = ChatPromptTemplate.from_template(
                    """ We know we need to increase the classifiers for our AI system. The classifiers are {modulators} The query is: {query}. Which of the classifiers should we decrease? Return just the modulator and desired value"""
                )
                chain_modulator = prompt_classify | self.llm
                classifier_output = await chain_modulator.ainvoke({"query": lookup_value_episodic, "modulators": str(attention_modulators)})
                print("classifier output 1", classifier_output)
                diff_layer = DifferentiableLayer(attention_modulators)
                adjusted_modulator = await diff_layer.adjust_weights(classifier_output)
                _input = prompt.format_prompt(query=adjusted_modulator)
                document_context_result = self.llm_base(_input.to_string())
                document_context_result_parsed = parser.parse(document_context_result)
                print("Updating with the following weights", str(document_context_result_parsed))
                await self.add_memories(observation=str(document_context_result_parsed), params=params, namespace="BUFFERMEMORY")
            else:
                # adjust the weights of the modulators by adding a negative value
                print("Lookup value, episodic", lookup_value_episodic)
                prompt_classify = ChatPromptTemplate.from_template(
                    """ We know we need to decrease the classifiers for our AI system. The classifiers are {modulators} The query is: {query}. Which of the classifiers should we decrease? Return just the modulator and desired value"""
                )
                chain_modulator_reduction = prompt_classify | self.llm

                classifier_output = await chain_modulator_reduction.ainvoke({"query": lookup_value_episodic, "modulators": str(attention_modulators)})
                print("classifier output 2", classifier_output)
                diff_layer = DifferentiableLayer(attention_modulators)
                adjusted_modulator =diff_layer.adjust_weights(classifier_output)
                _input = prompt.format_prompt(query=adjusted_modulator)
                document_context_result = self.llm_base(_input.to_string())
                document_context_result_parsed = parser.parse(document_context_result)
                print("Updating with the following weights", str(document_context_result_parsed))
                await self.add_memories(observation=str(document_context_result_parsed), params=params, namespace="BUFFERMEMORY")
            # except:
            #     # initialize the modulators with default values if they are not provided
            #     print("Starting with default modulators")
            #     attention_modulators = {
            #         "freshness": 0.5,
            #         "frequency": 0.5,
            #         "relevance": 0.5,
            #         "saliency": 0.5,
            #     }
            #     _input = prompt.format_prompt(query=attention_modulators)
            #     document_context_result = self.llm_base(_input.to_string())
            #     document_context_result_parsed = parser.parse(document_context_result)
            #     await self.add_memories(observation=str(document_context_result_parsed), params=params, namespace="BUFFERMEMORY")

        elif attention_modulators:
            pass


        lookup_value_semantic = await self.fetch_memories(
            observation=str(output), namespace="SEMANTICMEMORY"
        )
        print("This is the lookup value semantic", len(lookup_value_semantic))
        context = []
        memory_scores = []

        async def compute_score_for_memory(memory, output, attention_modulators):
            modulators = list(attention_modulators.keys())
            total_score = 0
            num_scores = 0
            individual_scores = {}  # Store individual scores with their modulator names

            for modulator in modulators:
                result = await self.handle_modulator(
                    modulator_name=modulator,
                    attention_modulators=attention_modulators,
                    observation=str(output),
                    namespace="EPISODICMEMORY",
                    memory=memory,
                )
                if result:
                    score = float(result[0])  # Assuming the first value in result is the score
                    individual_scores[modulator] = score  # Store the score with its modulator name
                    total_score += score
                    num_scores += 1

            average_score = total_score / num_scores if num_scores else 0
            return {
                "memory": memory,
                "average_score": average_score,
                "individual_scores": individual_scores
            }

        tasks = [
            compute_score_for_memory(memory=memory, output=output, attention_modulators=attention_modulators)
            for memory in lookup_value_semantic["data"]["Get"]["SEMANTICMEMORY"]
        ]

        memory_scores = await asyncio.gather(*tasks)
        # Sort the memories based on their average scores
        sorted_memories = sorted(memory_scores, key=lambda x: x["average_score"], reverse=True)[:5]
        # Store the sorted memories in the context
        context.extend([item for item in sorted_memories])

        for item in context:
            memory = item.get('memory', {})
            text = memory.get('text', '')

            prompt_sum= ChatPromptTemplate.from_template("""Based on this query: {query} Summarize the following text so it can be best used as a context summary for the user when running query: {text}"""
            )
            chain_sum = prompt_sum | self.llm
            summary_context = await chain_sum.ainvoke({"query": output, "text": text})
            item['memory']['text'] = summary_context


        print("HERE IS THE CONTEXT", context)

        lookup_value_episodic = await self.fetch_memories(
            observation=str(output), namespace="EPISODICMEMORY"
        )

        class Event(BaseModel):
            """Schema for an individual event."""

            event_order: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            event_name: str = Field(
                None, description="The task that needs to be performed"
            )
            operation: str = Field(None, description="The operation that was performed")
            original_query: str = Field(
                None, description="Original user query provided"
            )
        class EventList(BaseModel):
            """Schema for the record containing a list of events of the user chronologically."""

            tasks: List[Event] = Field(..., description="List of tasks")

        prompt_filter_chunk = f" Based on available memories {lookup_value_episodic} determine only the relevant list of steps and operations sequentially "
        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for determining what happened in the past and ordering events chronologically."
            ),
            HumanMessage(content="Analyze the following memories and provide the relevant response:"),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(content="Tips: Make sure to answer in the correct format"),
            HumanMessage(
                content="Tips: Only choose actions that are relevant to the user query and ignore others"
            )
        ]
        prompt_ = ChatPromptTemplate(messages=prompt_msgs)
        chain = create_structured_output_chain(
            EventList, self.llm, prompt_, verbose=True
        )
        from langchain.callbacks import get_openai_callback

        with get_openai_callback() as cb:
            episodic_context = await chain.arun(input=prompt_filter_chunk, verbose=True)
            print(cb)

        print("HERE IS THE EPISODIC CONTEXT", episodic_context)

        class BufferModulators(BaseModel):
            attention_modulators: Dict[str, float] = Field(... , description="Attention modulators")

        class BufferRawContextTerms(BaseModel):
            """Schema for documentGroups"""

            semantic_search_term: str = Field(
                ...,
                description="The search term to use to get relevant input based on user query",
            )
            document_content: str = Field(
                None, description="Shortened original content of the document"
            )
            attention_modulators_list: List[BufferModulators] = Field(
                ..., description="List of modulators"
            )
            average_modulator_score: str = Field(None, description="Average modulator score")
        class StructuredEpisodicEvents(BaseModel):
            """Schema for documentGroups"""

            event_order: str = Field(
                ...,
                description="Order when event occured",
            )
            event_type: str = Field(
                None, description="Type of the event"
            )
            event_context: List[BufferModulators] = Field(
                ..., description="Context of the event"
            )

        class BufferRawContextList(BaseModel):
            """Buffer raw context processed by the buffer"""

            docs: List[BufferRawContextTerms] = Field(..., description="List of docs")
            events: List[StructuredEpisodicEvents] = Field(..., description="List of events")
            user_query: str = Field(..., description="The original user query")

        # we structure the data here to make it easier to work with
        parser = PydanticOutputParser(pydantic_object=BufferRawContextList)
        prompt = PromptTemplate(
            template="""Summarize and create semantic search queries and relevant 
                        document summaries for the user query.\n
                        {format_instructions}\nOriginal query is: 
                        {query}\n Retrieved document context is: {context}. Retrieved memory context is {memory_context}""",
            input_variables=["query", "context", "memory_context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        _input = prompt.format_prompt(query=user_input, context=str(context), memory_context=str(episodic_context))
        document_context_result = self.llm_base(_input.to_string())
        document_context_result_parsed = parser.parse(document_context_result)
        # print(document_context_result_parsed)
        return document_context_result_parsed

    async def get_task_list(
        self, user_input=None, params=None, attention_modulators=None,
    ):
        """Gets the task list from the document context result to enchance it and be able to pass to the agent"""
        list_of_operations = await self.available_operations()

        class Task(BaseModel):
            """Schema for an individual task."""

            task_order: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            task_name: str = Field(
                None, description="The task that needs to be performed"
            )
            operation: str = Field(None, description="The operation to be performed")
            original_query: str = Field(
                None, description="Original user query provided"
            )

        class TaskList(BaseModel):
            """Schema for the record containing a list of tasks."""

            tasks: List[Task] = Field(..., description="List of tasks")

        prompt_filter_chunk = f" Based on available operations {list_of_operations} determine only the relevant list of steps and operations sequentially based {user_input}"
        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for decomposing prompts into steps and operations and choosing relevant ones"
            ),
            HumanMessage(content="Decompose based on the following prompt and provide relevant document context reponse:"),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(content="Tips: Make sure to answer in the correct format"),
            HumanMessage(
                content="Tips: Only choose actions that are relevant to the user query and ignore others"
            )
        ]
        prompt_ = ChatPromptTemplate(messages=prompt_msgs)
        chain = create_structured_output_chain(
            TaskList, self.llm, prompt_, verbose=True
        )
        from langchain.callbacks import get_openai_callback

        with get_openai_callback() as cb:
            output = await chain.arun(input=prompt_filter_chunk, verbose=True)
            print(cb)
        # output = json.dumps(output)
        my_object = parse_obj_as(TaskList, output)
        print("HERE IS THE OUTPUT", my_object.json())
        data = json.loads(my_object.json())
        # Extract the list of tasks
        tasks_list = data["tasks"]
        return tasks_list

    async def main_buffer(
        self, user_input=None, params=None, attention_modulators=None
    ):
        """AI buffer to run the AI agent to execute the set of tasks"""

        document_context_result_parsed = await self.buffer_context(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators,
        )
        tasks_list = await self.get_task_list(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators
        )
        result_tasks = []
        document_context_result_parsed = document_context_result_parsed.dict()
        document_from_vectorstore = [doc["document_content"] for doc in document_context_result_parsed["docs"]]

        for task in tasks_list:
            print("HERE IS THE TASK", task)

            complete_agent_prompt= f" Document context is: {document_from_vectorstore} \n  Task is : {task['task_order']} {task['task_name']} {task['operation']} "

            # task['vector_store_context_results']=document_context_result_parsed.dict()

            class FetchText(BaseModel):
                observation: str = Field(description="observation we want to translate")
            @tool("fetch_from_vector_store", args_schema=FetchText, return_direct=True)
            def fetch_from_vector_store(observation, args_schema=FetchText):
                """Fetch from vectorstore if data doesn't exist in the context"""
                if  document_context_result_parsed:
                    return document_context_result_parsed
                else:
                    out = self.fetch_memories(observation['original_query'], namespace="SEMANTICMEMORY")
                    return out

            class TranslateText(BaseModel):
                observation: str = Field(description="observation we want to translate")

            @tool("translate_to_de", args_schema=TranslateText, return_direct=True)
            def translate_to_de(observation, args_schema=TranslateText):
                """Translate to English"""
                out = GoogleTranslator(source="auto", target="de").translate(
                    text=observation
                )
                return out

            agent = initialize_agent(
                llm=self.llm,
                tools=[fetch_from_vector_store,translate_to_de],
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
            )

            output = agent.run(input=complete_agent_prompt )

            result_tasks.append(task)
            result_tasks.append(output)

        # buffer_result = await self.fetch_memories(observation=str(user_input))
        class EpisodicTask(BaseModel):
            """Schema for an individual task."""

            task_order: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            task_name: str = Field(
                None, description="The task that needs to be performed"
            )
            operation: str = Field(None, description="The operation to be performed")
            operation_result: str = Field(
                None, description="The result of the operation"
            )

        class EpisodicList(BaseModel):
            """Schema for the record containing a list of tasks."""

            tasks: List[EpisodicTask] = Field(..., description="List of tasks")
            start_date: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            end_date: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            user_query: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            attention_modulators: str = Field(..., description="List of attention modulators")
        parser = PydanticOutputParser(pydantic_object=EpisodicList)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        prompt = PromptTemplate(
            template="Format the result.\n{format_instructions}\nOriginal query is: {query}\n Steps are: {steps}, buffer is: {buffer}, date is:{date}, attention modulators are: {attention_modulators} \n",
            input_variables=["query", "steps", "buffer", "date", "attention_modulators"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        _input = prompt.format_prompt(
            query=user_input, steps=str(tasks_list)
            , buffer=str(result_tasks), date= date, attention_modulators=attention_modulators
        )
        print("HERE ARE THE STEPS, BUFFER AND DATE", str(tasks_list))
        print("here are the result_tasks", str(result_tasks))
        # return "a few things to do like load episodic memory in a structured format"
        output = self.llm_base(_input.to_string())
        result_parsing = parser.parse(output)
        lookup_value = await self.add_memories(
            observation=str(result_parsing.json()), params=params, namespace='EPISODICMEMORY'
        )
        # await self.delete_memories()
        return result_parsing.json()


class LongTermMemory:
    def __init__(
        self,
        user_id: str = "676",
        memory_id: Optional[str] = None,
        index_name: Optional[str] = None,
        db_type: str = "weaviate",
    ):
        self.user_id = user_id
        self.memory_id = memory_id
        self.ltm_memory_id = str(uuid.uuid4())
        self.index_name = index_name
        self.db_type = db_type
        self.semantic_memory = SemanticMemory(user_id, memory_id, index_name, db_type)
        self.episodic_memory = EpisodicMemory(user_id, memory_id, index_name, db_type)


class ShortTermMemory:
    def __init__(
        self,
        user_id: str = "676",
        memory_id: Optional[str] = None,
        index_name: Optional[str] = None,
        db_type: str = "weaviate",
    ):
        self.user_id = user_id
        self.memory_id = memory_id
        self.stm_memory_id = str(uuid.uuid4())
        self.index_name = index_name
        self.db_type = db_type
        self.episodic_buffer = EpisodicBuffer(user_id, memory_id, index_name, db_type)


class Memory:
    load_dotenv()
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    def __init__(
        self,
        user_id: str = "676",
        index_name: str = None,
        knowledge_source: str = None,
        knowledge_type: str = None,
        db_type: str = "weaviate",
        namespace: str = None,
    ) -> None:
        self.user_id = user_id
        self.index_name = index_name
        self.db_type = db_type
        self.knowledge_source = knowledge_source
        self.knowledge_type = knowledge_type
        self.memory_id = str(uuid.uuid4())
        self.long_term_memory = None
        self.short_term_memory = None
        self.namespace = namespace
        load_dotenv()

    # Asynchronous factory function for creating LongTermMemory
    async def async_create_long_term_memory(
        self, user_id, memory_id, index_name, db_type
    ):
        # Perform asynchronous initialization steps if needed
        return LongTermMemory(
            user_id=self.user_id,
            memory_id=self.memory_id,
            index_name=self.index_name,
            db_type=self.db_type,
        )

    async def async_init(self):
        # Asynchronous initialization of LongTermMemory and ShortTermMemory
        self.long_term_memory = await self.async_create_long_term_memory(
            user_id=self.user_id,
            memory_id=self.memory_id,
            index_name=self.index_name,
            db_type=self.db_type,
        )
        self.short_term_memory = await self.async_create_short_term_memory(
            user_id=self.user_id,
            memory_id=self.memory_id,
            index_name=self.index_name,
            db_type=self.db_type,
        )

    async def async_create_short_term_memory(
        self, user_id, memory_id, index_name, db_type
    ):
        # Perform asynchronous initialization steps if needed
        return ShortTermMemory(
            user_id=self.user_id,
            memory_id=self.memory_id,
            index_name=self.index_name,
            db_type=self.db_type,
        )

    async def _add_semantic_memory(
        self, observation: str, loader_settings: dict = None, params: dict = None
    ):
        return await self.long_term_memory.semantic_memory.add_memories(
            observation=observation,
            loader_settings=loader_settings,
            params=params,
        )

    async def _fetch_semantic_memory(self, observation, params):
        return await self.long_term_memory.semantic_memory.fetch_memories(
            observation=observation, params=params
        )

    async def _delete_semantic_memory(self, params: str = None):
        return await self.long_term_memory.semantic_memory.delete_memories(
            params=params
        )

    async def _add_episodic_memory(
        self, observation: str, loader_settings: dict = None, params: dict = None
    ):
        return await self.long_term_memory.episodic_memory.add_memories(
            observation=observation, loader_settings=loader_settings, params=params
        )

    async def _fetch_episodic_memory(self, observation, params: str = None):
        return await self.long_term_memory.episodic_memory.fetch_memories(
            observation=observation, params=params
        )

    async def _delete_episodic_memory(self, params: str = None):
        return await self.long_term_memory.episodic_memory.delete_memories(
            params=params
        )


    async def _add_buffer_memory(
        self,
        user_input: str,
        namespace: str = None,
        loader_settings: dict = None,
        params: dict = None,
    ):
        return await self.short_term_memory.episodic_buffer.add_memories(
            observation=user_input, loader_settings=loader_settings, params=params
        )

    async def _fetch_buffer_memory(self, user_input: str):
        return await self.short_term_memory.episodic_buffer.fetch_memories(
            observation=user_input
        )

    async def _delete_buffer_memory(self, params: str = None):
        return await self.short_term_memory.episodic_buffer.delete_memories(
            params=params
        )

    async def _create_buffer_context(
        self,
        user_input: str,
        params: dict = None,
        attention_modulators: dict = None,
    ):
        return await self.short_term_memory.episodic_buffer.buffer_context(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators,
        )
    async def _get_task_list(
        self,
        user_input: str,
        params: str = None,
        attention_modulators: dict = None,
    ):
        return await self.short_term_memory.episodic_buffer.get_task_list(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators,
        )
    async def _run_main_buffer(
        self,
        user_input: str,
        params: dict = None,
        attention_modulators: dict = None,
    ):
        return await self.short_term_memory.episodic_buffer.main_buffer(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators,
        )

    async def _available_operations(self):
        return await self.long_term_memory.episodic_buffer.available_operations()

    async def _provide_feedback(self, score:str =None, params: dict = None, attention_modulators: dict = None):
        return await self.short_term_memory.episodic_buffer.provide_feedback(score=score, params=params, attention_modulators=attention_modulators)


async def main():

    # if you want to run the script as a standalone script, do so with the examples below
    memory = Memory(user_id="TestUser")
    await memory.async_init()
    params = {
        "version": "1.0",
        "agreement_id": "AG123456",
        "privacy_policy": "https://example.com/privacy",
        "terms_of_service": "https://example.com/terms",
        "format": "json",
        "schema_version": "1.1",
        "checksum": "a1b2c3d4e5f6",
        "owner": "John Doe",
        "license": "MIT",
        "validity_start": "2023-08-01",
        "validity_end": "2024-07-31",
    }
    loader_settings =  {
    "format": "PDF",
    "source": "url",
    "path": "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"
    }
    load_jack_london = await memory._add_semantic_memory(observation = "bla", loader_settings=loader_settings, params=params)
    print(load_jack_london)

    modulator = {"relevance": 0.1,  "frequency": 0.1}

    # fdsf = await memory._fetch_semantic_memory(observation="bla", params=None)
    # print(fdsf)
    # await memory._delete_episodic_memory()
    #
    # run_main_buffer = await memory._create_buffer_context(
    #     user_input="I want to know how does Buck adapt to life in the wild and then have that info translated to german ",
    #     params=params,
    #     attention_modulators=modulator,
    # )
    # print(run_main_buffer)
    # #
    # run_main_buffer = await memory._run_main_buffer(
    #     user_input="I want to know how does Buck adapt to life in the wild and then have that info translated to german ",
    #     params=params,
    #     attention_modulators=None,
    # )
    # print(run_main_buffer)
    # del_semantic = await memory._delete_semantic_memory()
    # print(del_semantic)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


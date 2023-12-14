
import json
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
import marvin
from deep_translator import GoogleTranslator

from langchain.agents import initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import tool
from pydantic import parse_obj_as

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from typing import Optional, Dict, List, Union

import tracemalloc


import os
from datetime import datetime
from langchain import PromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
import uuid

load_dotenv()




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
        description: str = Field(..., description="Latest buffer modulators")
        direction: str = Field(..., description="Increase or a decrease of the modulator")

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
        chain_filter = prompt_classify | self.llm.bind(function_call={"name": "classifier"}, functions=json_structure)
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
            classifier_output = await chain_modulator.ainvoke(
                {"query": lookup_value_episodic, "modulators": str(attention_modulators)})
            print("classifier output 1", classifier_output)
            diff_layer = DifferentiableLayer(attention_modulators)
            adjusted_modulator = await diff_layer.adjust_weights(classifier_output)
            _input = prompt.format_prompt(query=adjusted_modulator)
            document_context_result = self.llm_base(_input.to_string())
            document_context_result_parsed = parser.parse(document_context_result)
            print("Updating with the following weights", str(document_context_result_parsed))
            await self.add_memories(observation=str(document_context_result_parsed), params=params,
                                    namespace="BUFFERMEMORY")
        else:
            # adjust the weights of the modulators by adding a negative value
            print("Lookup value, episodic", lookup_value_episodic)
            prompt_classify = ChatPromptTemplate.from_template(
                """ We know we need to decrease the classifiers for our AI system. The classifiers are {modulators} The query is: {query}. Which of the classifiers should we decrease? Return just the modulator and desired value"""
            )
            chain_modulator_reduction = prompt_classify | self.llm

            classifier_output = await chain_modulator_reduction.ainvoke(
                {"query": lookup_value_episodic, "modulators": str(attention_modulators)})
            print("classifier output 2", classifier_output)
            diff_layer = DifferentiableLayer(attention_modulators)
            adjusted_modulator = diff_layer.adjust_weights(classifier_output)
            _input = prompt.format_prompt(query=adjusted_modulator)
            document_context_result = self.llm_base(_input.to_string())
            document_context_result_parsed = parser.parse(document_context_result)
            print("Updating with the following weights", str(document_context_result_parsed))
            await self.add_memories(observation=str(document_context_result_parsed), params=params,
                                    namespace="BUFFERMEMORY")
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

        prompt_sum = ChatPromptTemplate.from_template(
            """Based on this query: {query} Summarize the following text so it can be best used as a context summary for the user when running query: {text}"""
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
        attention_modulators: Dict[str, float] = Field(..., description="Attention modulators")

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

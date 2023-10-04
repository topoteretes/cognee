# Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client
import json
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
import marvin
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import tool
from pydantic import parse_obj_as

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
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
import uuid

load_dotenv()



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

        complete_agent_prompt = f" Document context is: {document_from_vectorstore} \n  Task is : {task['task_order']} {task['task_name']} {task['operation']} "

        # task['vector_store_context_results']=document_context_result_parsed.dict()

        class FetchText(BaseModel):
            observation: str = Field(description="observation we want to translate")

        @tool("fetch_from_vector_store", args_schema=FetchText, return_direct=True)
        def fetch_from_vector_store(observation, args_schema=FetchText):
            """Fetch from vectorstore if data doesn't exist in the context"""
            if document_context_result_parsed:
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
            tools=[fetch_from_vector_store, translate_to_de],
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
        )

        output = agent.run(input=complete_agent_prompt)

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
        , buffer=str(result_tasks), date=date, attention_modulators=attention_modulators
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
import os
from datetime import datetime
from typing import List

from langchain import PromptTemplate, OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import dotenv
from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
dotenv.load_dotenv()

llm_base = OpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4-0613",
        )
async def _add_to_episodic(user_input, tasks_list, result_tasks, attention_modulators, params):


    memory = Memory(user_id="TestUser")
    await memory.async_init()
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

    # return "a few things to do like load episodic memory in a structured format"
    output = llm_base(_input.to_string())
    result_parsing = parser.parse(output)
    lookup_value = await memory._add_episodic_memory(
        observation=str(result_parsing.json()), params=params
    )


async def add_to_buffer(adjusted_modulator=None, params={}):
    memory = Memory(user_id="TestUser")
    await memory.async_init()
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
    _input = prompt.format_prompt(query=adjusted_modulator)
    document_context_result = llm_base(_input.to_string())
    document_context_result_parsed = parser.parse(document_context_result)
    await memory._add_buffer_memory(user_input=str(document_context_result_parsed), params=params)
    return document_context_result_parsed.json()


async def delete_from_buffer():
    from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
    memory = Memory(user_id="TestUser")
    await memory.async_init()
    await memory._delete_buffer_memory()

async def delete_from_episodic():
    from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
    memory = Memory(user_id="TestUser")
    await memory.async_init()
    await memory._delete_episodic_memory()


async def get_from_episodic(observation=None):
    from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
    memory = Memory(user_id="TestUser")
    await memory.async_init()
    return await memory._fetch_episodic_memory(observation=observation)

async def get_from_buffer(observation=None):
    from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
    memory = Memory(user_id="TestUser")
    await memory.async_init()
    return await memory._fetch_buffer_memory(user_input=observation)


async def main():
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
    modulator = {"relevance": 1.0, "saliency": 1.0, "frequency": 1.0, "freshness": 1.0, "repetition": 1.0}
    user_input = "I want to know how does Buck adapt to life in the wild"
    # tasks_list = """tasks": [{"task_order": "1", "task_name": "Fetch Information", "operation": "fetch from vector store", "original_query": "I want to know how does Buck adapt to life in the wild"]"""
    out_tasks = """here are the result_tasks [{'task_order': '1', 'task_name': 'Save Information', 'operation': 'save to vector store', 'original_query': 'Add to notes who is Buck and get info saved yesterday about him'}, {'docs': [{'semantic_search_term': "Add to notes who is Buck", 'document_summary': 'Buck was a dog stolen from his home', 'document_relevance': '0.75', 'attention_modulators_list': [{'frequency': '0.33', 'saliency': '0.75', 'relevance': '0.74'}]}], 'user_query': 'I want to know who buck is and check my notes from yesterday'}, {'task_order': '2', 'task_name': 'Check historical data', 'operation': 'check historical data', 'original_query': ' check my notes from yesterday'}, ' Data saved yesterday about Buck include informaton that he was stolen from home and that he was a pretty dog ']"""

    await _add_to_episodic(user_input=user_input, result_tasks=out_tasks, tasks_list=None, attention_modulators=modulator, params=params)
    # await delete_from_episodic()
    # aa = await get_from_episodic(observation="summary")
    # await delete_from_buffer()
    modulator_changed = {"relevance": 0.9, "saliency": 0.9, "frequency": 0.9}
    await add_to_buffer(adjusted_modulator=modulator_changed)

    # aa = await get_from_buffer(observation="summary")
    # print(aa)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



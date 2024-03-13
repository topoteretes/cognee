""" This module contains the classifiers for the documents. """

import json
import logging

from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

from ..config import Config

config = Config()
config.load()
OPENAI_API_KEY = config.openai_key

async def classify_documents(query: str, document_id: str, content: str):
    """Classify the documents based on the query and content."""
    document_context = content
    logging.info("This is the document context %s", document_context)

    llm = ChatOpenAI(temperature=0, model=config.model)
    prompt_classify = ChatPromptTemplate.from_template(
        """You are a summarizer and classifier.
         Determine what book this is and where does it belong in the output :
          {query}, Id: {d_id} Document context is: {context}"""
    )
    json_structure = [
        {
            "name": "summarizer",
            "description": "Summarization and classification",
            "parameters": {
                "type": "object",
                "properties": {
                    "DocumentCategory": {
                        "type": "string",
                        "description": "The classification of documents "
                                       "in groups such as legal, medical, etc.",
                    },
                    "Title": {
                        "type": "string",
                        "description": "The title of the document",
                    },
                    "Summary": {
                        "type": "string",
                        "description": "The summary of the document",
                    },
                    "d_id": {"type": "string", "description": "The id of the document"},
                },
                "required": ["DocumentCategory", "Title", "Summary", "d_id"],
            },
        }
    ]
    chain_filter = prompt_classify | llm.bind(
        function_call={"name": "summarizer"}, functions=json_structure
    )
    classifier_output = await chain_filter.ainvoke(
        {"query": query, "d_id": document_id, "context": str(document_context)}
    )

    arguments_str = classifier_output.additional_kwargs["function_call"]["arguments"]
    logging.info("This is the arguments string %s", arguments_str)
    arguments_dict = json.loads(arguments_str)
    return arguments_dict

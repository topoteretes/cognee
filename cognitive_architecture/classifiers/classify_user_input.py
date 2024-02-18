""" This module contains the classifiers for the documents. """

import logging

from langchain.prompts import ChatPromptTemplate
import json


from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

from ..config import Config
from ..database.vectordb.loaders.loaders import _document_loader

config = Config()
config.load()
OPENAI_API_KEY = config.openai_key

async def classify_user_input(query, input_type):
    """ Classify the user input based on the query and input type."""
    llm = ChatOpenAI(temperature=0, model=config.model)
    prompt_classify = ChatPromptTemplate.from_template(
        """You are a  classifier. 
        Determine with a True or False if the following input: {query}, 
        is relevant for the following memory category: {input_type}"""
    )
    json_structure = [
        {
            "name": "classifier",
            "description": "Classification",
            "parameters": {
                "type": "object",
                "properties": {
                    "InputClassification": {
                        "type": "boolean",
                        "description": "The classification of the input",
                    }
                },
                "required": ["InputClassification"],
            },
        }
    ]
    chain_filter = prompt_classify | llm.bind(
        function_call={"name": "classifier"}, functions=json_structure
    )
    classifier_output = await chain_filter.ainvoke(
        {"query": query, "input_type": input_type}
    )
    arguments_str = classifier_output.additional_kwargs["function_call"]["arguments"]
    logging.info("This is the arguments string %s", arguments_str)
    arguments_dict = json.loads(arguments_str)
    logging.info("Relevant summary is %s", arguments_dict.get("DocumentSummary", None))
    InputClassification = arguments_dict.get("InputClassification", None)
    logging.info("This is the classification %s", InputClassification)
    return InputClassification
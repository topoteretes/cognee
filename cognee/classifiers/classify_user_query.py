""" This module contains the function to classify the user query. """

import json

from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

from ..config import Config

config = Config()
config.load()
OPENAI_API_KEY = config.openai_key

async def classify_user_query(query, context, document_types):
    """Classify the user query based on the context and document types."""
    llm = ChatOpenAI(temperature=0, model=config.model)
    prompt_classify = ChatPromptTemplate.from_template(
        """You are a  classifier. 
        You store user memories, thoughts and feelings. 
        Determine if you need to use them to answer this query : {query}"""
    )
    json_structure = [
        {
            "name": "classifier",
            "description": "Classification",
            "parameters": {
                "type": "object",
                "properties": {
                    "UserQueryClassifier": {
                        "type": "bool",
                        "description": "The classification of documents "
                                       "in groups such as legal, medical, etc.",
                    }
                },
                "required": ["UserQueryClassifier"],
            },
        }
    ]
    chain_filter = prompt_classify | llm.bind(
        function_call={"name": "classifier"}, functions=json_structure
    )
    classifier_output = await chain_filter.ainvoke(
        {"query": query, "context": context, "document_types": document_types}
    )
    arguments_str = classifier_output.additional_kwargs["function_call"]["arguments"]
    print("This is the arguments string", arguments_str)
    arguments_dict = json.loads(arguments_str)
    classfier_value = arguments_dict.get("UserQueryClassifier", None)
    print("This is the classifier value", classfier_value)
    return classfier_value

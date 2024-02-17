import logging

from langchain.prompts import ChatPromptTemplate
import json

# TO DO, ADD ALL CLASSIFIERS HERE


from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

from ..config import Config
from ..database.vectordb.loaders.loaders import _document_loader

config = Config()
config.load()
OPENAI_API_KEY = config.openai_key
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader






async def classify_summary(query, document_summaries):
    llm = ChatOpenAI(temperature=0, model=config.model)
    prompt_classify = ChatPromptTemplate.from_template(
        """You are a  classifier. Determine what document  are relevant for the given query: {query}, Document summaries and ids:{document_summaries}"""
    )
    json_structure = [
        {
            "name": "classifier",
            "description": "Classification",
            "parameters": {
                "type": "object",
                "properties": {
                    "DocumentSummary": {
                        "type": "string",
                        "description": "The summary of the document and the topic it deals with.",
                    },
                    "d_id": {"type": "string", "description": "The id of the document"},
                },
                "required": ["DocumentSummary"],
            },
        }
    ]
    chain_filter = prompt_classify | llm.bind(
        function_call={"name": "classifier"}, functions=json_structure
    )
    classifier_output = await chain_filter.ainvoke(
        {"query": query, "document_summaries": document_summaries}
    )
    arguments_str = classifier_output.additional_kwargs["function_call"]["arguments"]
    print("This is the arguments string", arguments_str)
    arguments_dict = json.loads(arguments_str)
    logging.info("Relevant summary is %s", arguments_dict.get("DocumentSummary", None))
    classfier_id = arguments_dict.get("d_id", None)

    print("This is the classifier id ", classfier_id)

    return classfier_id
import logging

from langchain.prompts import ChatPromptTemplate
import json

#TO DO, ADD ALL CLASSIFIERS HERE







from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

from ..config import Config
from ..database.vectordb.loaders.loaders import _document_loader
config = Config()
config.load()
OPENAI_API_KEY = config.openai_key
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader


async def classify_documents(query:str, document_id:str, content:str):

    document_context  = content
    logging.info("This is the document context", document_context)

    llm = ChatOpenAI(temperature=0, model=config.model)
    prompt_classify = ChatPromptTemplate.from_template(
        """You are a summarizer and classifier. Determine what book this is and where does it belong in the output : {query}, Id: {d_id} Document context is: {context}"""
    )
    json_structure = [{
        "name": "summarizer",
        "description": "Summarization and classification",
        "parameters": {
            "type": "object",
            "properties": {
                "DocumentCategory": {
                    "type": "string",
                    "description": "The classification of documents in groups such as legal, medical, etc."
                },
                "Title": {
                    "type": "string",
                    "description": "The title of the document"
                },
                "Summary": {
                    "type": "string",
                    "description": "The summary of the document"
                },
                "d_id": {
                    "type": "string",
                    "description": "The id of the document"
                }


            }, "required": ["DocumentCategory", "Title", "Summary","d_id"] }
    }]
    chain_filter = prompt_classify | llm.bind(function_call={"name": "summarizer"}, functions=json_structure)
    classifier_output = await chain_filter.ainvoke({"query": query, "d_id": document_id, "context": str(document_context)})
    arguments_str = classifier_output.additional_kwargs['function_call']['arguments']
    print("This is the arguments string", arguments_str)
    arguments_dict = json.loads(arguments_str)
    return arguments_dict



# classify retrievals according to type of retrieval
def classify_retrieval():
    pass


# classify documents according to type of document
async def classify_call(query, document_summaries):

    llm = ChatOpenAI(temperature=0, model=config.model)
    prompt_classify = ChatPromptTemplate.from_template(
        """You are a  classifier. Determine what document  are relevant for the given query: {query}, Document summaries:{document_summaries}"""
    )
    json_structure = [{
        "name": "classifier",
        "description": "Classification",
        "parameters": {
            "type": "object",
            "properties": {
                "DocumentSummary": {
                    "type": "string",
                    "description": "The summary of the document and the topic it deals with."
                }


            }, "required": ["DocumentSummary"] }
    }]
    chain_filter = prompt_classify | llm.bind(function_call={"name": "classifier"}, functions=json_structure)
    classifier_output = await chain_filter.ainvoke({"query": query, "document_summaries": document_summaries})
    arguments_str = classifier_output.additional_kwargs['function_call']['arguments']
    print("This is the arguments string", arguments_str)
    arguments_dict = json.loads(arguments_str)
    classfier_value = arguments_dict.get('DocumentSummary', None)

    print("This is the classifier value", classfier_value)

    return classfier_value



async def classify_user_query(query, context, document_types):

    llm = ChatOpenAI(temperature=0, model=config.model)
    prompt_classify = ChatPromptTemplate.from_template(
        """You are a  classifier. You store user memories, thoughts and feelings. Determine if you need to use them to answer this query : {query}"""
    )
    json_structure = [{
        "name": "classifier",
        "description": "Classification",
        "parameters": {
            "type": "object",
            "properties": {
                "UserQueryClassifier": {
                    "type": "bool",
                    "description": "The classification of documents in groups such as legal, medical, etc."
                }


            }, "required": ["UserQueryClassiffier"] }
    }]
    chain_filter = prompt_classify | llm.bind(function_call={"name": "classifier"}, functions=json_structure)
    classifier_output = await chain_filter.ainvoke({"query": query, "context": context, "document_types": document_types})
    arguments_str = classifier_output.additional_kwargs['function_call']['arguments']
    print("This is the arguments string", arguments_str)
    arguments_dict = json.loads(arguments_str)
    classfier_value = arguments_dict.get('UserQueryClassifier', None)

    print("This is the classifier value", classfier_value)

    return classfier_value
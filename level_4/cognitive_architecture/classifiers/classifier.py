from langchain.prompts import ChatPromptTemplate
import json

#TO DO, ADD ALL CLASSIFIERS HERE







from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

from ..config import Config

config = Config()
config.load()
OPENAI_API_KEY = config.openai_key
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader


async def classify_documents(query):

    llm = ChatOpenAI(temperature=0, model=config.model)
    prompt_classify = ChatPromptTemplate.from_template(
        """You are a summarizer and classifier. Determine what book this is and where does it belong in the output : {query}"""
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
                }


            }, "required": ["DocumentCategory", "Title", "Summary"] }
    }]
    chain_filter = prompt_classify | llm.bind(function_call={"name": "summarizer"}, functions=json_structure)
    classifier_output = await chain_filter.ainvoke({"query": query})
    arguments_str = classifier_output.additional_kwargs['function_call']['arguments']
    print("This is the arguments string", arguments_str)
    arguments_dict = json.loads(arguments_str)
    classfier_value = arguments_dict.get('summarizer', None)

    print("This is the classifier value", classfier_value)

    return classfier_value



# classify retrievals according to type of retrieval
def classify_retrieval():
    pass


# classify documents according to type of document
def classify_call():
    pass
# QUICKSTART

!!! tip "To understand how cognee works check out the [conceptual overview](conceptual_overview.md)"

## Setup

To run cognee, you will need the following:

1. A running postgres instance
2. OpenAI API key (Ollama or Anyscale could work as [well](local_models.md))

Navigate to cognee folder and run
```
docker compose up postgres
```

Add your LLM API key to the environment variables

```
import os

os.environ["LLM_API_KEY"] = "YOUR_OPENAI_API_KEY"
```
or 
```
cognee.config.llm_api_key = "YOUR_OPENAI_API_KEY"

```
If you are using Networkx, create an account on Graphistry to vizualize results:
```
   
   cognee.config.set_graphistry_username = "YOUR_USERNAME"
   cognee.config.set_graphistry_password = "YOUR_PASSWORD"
```
## Run

cognee is asynchronous by design, meaning that operations like adding information, processing it, and querying it can run concurrently without blocking the execution of other tasks. 
Make sure to await the results of the functions that you call.

```
import cognee

text = """Natural language processing (NLP) is an interdisciplinary
       subfield of computer science and information retrieval"""

cognee.add(text) # Add a new piece of information

cognee.cognify() # Use LLMs and cognee to create knowledge

search_results = cognee.search("SIMILARITY", {'query': 'Tell me about NLP'}) # Query cognee for the knowledge

for result_text in search_results[0]:
    print(result_text)
```

In the example above, we add a piece of information to cognee, use LLMs to create a GraphRAG, and then query cognee for the knowledge.
cognee is composable and you can build your own cognee pipelines using our [templates.](templates.md)

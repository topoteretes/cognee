# QUICKSTART

!!! tip "To understand how cognee works check out the [conceptual overview](conceptual_overview.md)"

## Setup

To run cognee, you will need the following:

1. OpenAI API key (Ollama or Anyscale could work as [well](local_models.md))

Add your LLM API key to the environment variables

```
import os

os.environ["LLM_API_KEY"] = "YOUR_OPENAI_API_KEY"
```
or 
```
cognee.config.llm_api_key = "YOUR_OPENAI_API_KEY"

```
If you are using Networkx, create an account on Graphistry to visualize results:
```
    cognee.config.set_graphistry_config({
        "username": "YOUR_USERNAME",
        "password": "YOUR_PASSWORD"
    })
```

If you want to run Postgres instead of Sqlite, run postgres Docker container.
Navigate to cognee folder and run:
```
docker compose up postgres
```

Add the following environment variables to .env file
```
DB_HOST=127.0.0.1
DB_PORT=5432
DB_USERNAME=cognee # or any username you want
DB_PASSWORD=cognee # or any password you want
DB_NAME=cognee_db # or any db name you want
DB_PROVIDER=postgres
```

## Run

cognee is asynchronous by design, meaning that operations like adding information, processing it, and querying it can run concurrently without blocking the execution of other tasks. 
Make sure to await the results of the functions that you call.

```
import cognee

text = """Natural language processing (NLP) is an interdisciplinary
       subfield of computer science and information retrieval"""

await cognee.add(text) # Add a new piece of information

await cognee.cognify() # Use LLMs and cognee to create knowledge

search_results = await cognee.search(SearchType.INSIGHTS, query_text='Tell me about NLP')  # Query cognee for the knowledge

for result_text in search_results:
    print(result_text)
```

In the example above, we add a piece of information to cognee, use LLMs to create a GraphRAG, and then query cognee for the knowledge.
cognee is composable and you can build your own cognee pipelines using our [templates.](templates.md)

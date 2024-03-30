# cognee 


## Make data processing for LLMs easy


_Open-source framework for creating knowledge graphs and data models for LLMs._


---


[![Twitter Follow](https://img.shields.io/twitter/follow/tricalt?style=social)](https://twitter.com/tricalt)

[![Downloads](https://img.shields.io/pypi/dm/cognee.svg)](https://pypi.python.org/pypi/cognee)



cognee makes it easy to reliably enrich data for Large Language Models (LLMs) like GPT-3.5, GPT-4, GPT-4-Vision, including in the future the open source models like Mistral/Mixtral from Together, Anyscale, Ollama, and llama-cpp-python.

By leveraging various tools like graph databases, function calling, tool calling and Pydantic; cognee stands out for its aim to emulate human memory for LLM apps and frameworks. 

We leverage Neo4j to do the heavy lifting and dlt to load the data, and we've built a simple, easy-to-use API on top of it by helping you manage your context



## Getting Started

### Setup

Create `.env` file in your project root directory in order to store environment variables such as API keys.

Note: Don't push `.env` file to git repo as it will expose those keys to others.

If cognee is installed with Weaviate as a vector database provider, add Weaviate environment variables:
```
WEAVIATE_URL = "YOUR_WEAVIATE_URL"
WEAVIATE_API_KEY = "YOUR_WEAVIATE_API_KEY"
```

Otherwise if cognee is installed with a default (Qdrant) vector database provider, add Qdrant environment variables:
```
QDRANT_URL = "YOUR_QDRANT_URL"
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"
```

Add OpenAI API Key environment variable:
```
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```

Cognee stores data and system files inside the library directory, which is lost if the library folder is removed.
You can change the directories where cognee will store data and system files by calling config functions:
```
import cognee

cognee.config.system_root_directory(absolute_path_to_directory)

cognee.config.data_root_directory(absolute_path_to_directory)
```

#### Without .env file

```
import os

os.environ["WEAVIATE_URL"] = "YOUR_WEAVIATE_URL"
os.environ["WEAVIATE_API_KEY"] = "YOUR_WEAVIATE_API_KEY"

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

```

### Run

Add a new piece of information to the storage:
```
import cognee

cognee.add("some_text", dataset_name)

cognee.add([
    "some_text_1",
    "some_text_2",
    "some_text_3",
    ...
])
```
Or
```
cognee.add("file://{absolute_path_to_file}", dataset_name)

cognee.add(
    [
        "file://{absolute_path_to_file_1}",
        "file://{absolute_path_to_file_2}",
        "file://{absolute_path_to_file_3}",
        ...
    ],
    dataset_name
)
```
Or
```
cognee.add("data://{absolute_path_to_directory}", dataset_name)

# This is useful if you have a directory with files organized in subdirectories.
# You can target which directory to add by providing dataset_name.
# Example:
#            root
#           /    \
#      reports  bills
#     /       \
#   2024     2023
#
# cognee.add("data://{absolute_path_to_root}", "reports.2024")
# This will add just directory 2024 under reports.
```

Use LLMs and cognee to create graphs:
``` 
cognee.cognify(dataset_name)
 ``` 

Render the graph with our util function:

```
from cognee.utils import render_graph

graph_url = await render_graph(graph)

print(graph_url)
```

Query the graph for a piece of information:
```
search_results = cognee.search('SIMILARITY', "query_search")

print(search_results)
```


## Why use cognee?

The question of using cognee is fundamentally a question of why to structure data inputs and outputs for your llm workflows.

1. **Cost effective** — cognee extends the capabilities of your LLMs without the need for expensive data processing tools.

2. **Self contained** — cognee runs as a library and is simple to use

3. **Interpretable** — Navigate graphs instead of embeddings to understand your data.

4. **User Guided** cognee lets you control your input and provide your own Pydantic data models 



## License

This project is licensed under the terms of the Apache License 2.0.

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


```

pip install -U cognee

```
Set OpenAI API Key as an environment variable


```
import os

# Setting an environment variable
os.environ['OPENAI_API_KEY'] = ''


```

Import cognee and start using it


```
import cognee
from os import listdir, path
from cognee import add

data_path = path.abspath(".data")

results = await add(data_path, "izmene")
for result in results:
    print(result)
    
```

Run the following command to see the graph. 
Make sure to add your Graphistry credentials to .env beforehand

```
from cognee.utils import render_graph

graph = await cognee.cognify("izmene")
graph_url = await render_graph(graph, graph_type = "networkx")
print(graph_url)
```


Search the graph for a piece of information

```
from cognee import search
from cognee.api.v1.search.search import SearchType
query_params = {
    SearchType.SIMILARITY: {'query': 'your search query here'}
}
out = await search(graph, query_params)
```



[//]: # (You can also check out our [cookbook](./examples/index.md)  to learn more about how to use cognee.)



## Why use cognee?


The question of using cognee is fundamentally a question of why to structure data inputs and outputs for your llm workflows.


1. **Cost effective** — cognee extends the capabilities of your LLMs without the need for expensive data processing tools.


2. **Self contained** — cognee runs as a library and is simple to use


3. **Interpretable** — Navigate graphs instead of embeddings to understand your data.


4. **User Guided** cognee lets you control your input and provide your own Pydantic data models 


## License


This project is licensed under the terms of the MIT License.
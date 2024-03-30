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

pip install "cognee[weaviate]"

```
Set OpenAI API Key as an environment variable


```
import os

os.environ["WEAVIATE_URL"] = "YOUR_WEAVIATE_URL"
os.environ["WEAVIATE_API_KEY"] = "YOUR_WEAVIATE_API_KEY"

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


```

Import cognee and start using it


```
import cognee

text = """Natural language processing (NLP) is an interdisciplinary
       subfield of computer science and information retrieval"""

cognee.add(text) # Add a new piece of information

cognee.cognify() # Use LLMs and cognee to create knowledge

search_results = cognee.search("SIMILARITY", "computer science") # Query cognee for the knowledge

for result_text in search_results[0]:
    print(result_text)
    
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
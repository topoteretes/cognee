# QUICKSTART

!!! tip "To understand how cognee works check out the [conceptual overview](conceptual_overview.md)"

## Setup

You will need a Weaviate instance and an OpenAI API key to use cognee.
Weaviate let's you run an instance for 14 days for free. You can sign up at their website: [Weaviate](https://www.semi.technology/products/weaviate.html)


You can also use Ollama or Anyscale as your LLM provider. For more info on local models check [here](local_models.md)

```
import os

os.environ["WEAVIATE_URL"] = "YOUR_WEAVIATE_URL"
os.environ["WEAVIATE_API_KEY"] = "YOUR_WEAVIATE_API_KEY"

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```
## Run

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
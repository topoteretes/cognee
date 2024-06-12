# QUICKSTART

!!! tip "To understand how cognee works check out the [conceptual overview](conceptual_overview.md)"

## Setup

You will need a Weaviate instance and an OpenAI API key to use cognee.
Weaviate let's you run an instance for 14 days for free. You can sign up at their website: [Weaviate](https://www.semi.technology/products/weaviate.html)


You can also use Ollama or Anyscale as your LLM provider. For more info on local models check [here](local_models.md)

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
# QUICKSTART

Setup

```
import os

os.environ["WEAVIATE_URL"] = "YOUR_WEAVIATE_URL"
os.environ["WEAVIATE_API_KEY"] = "YOUR_WEAVIATE_API_KEY"

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```
Run

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
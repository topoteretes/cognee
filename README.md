# cognee
[![GitHub forks](https://img.shields.io/github/forks/topoteretes/cognee.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/network/)
[![GitHub stars](https://img.shields.io/github/stars/topoteretes/cognee.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/stargazers/)
[![GitHub commits](https://badgen.net/github/commits/topoteretes/cognee)](https://GitHub.com/topoteretes/cognee/commit/)
[![Github tag](https://badgen.net/github/tag/topoteretes/cognee)](https://github.com/topoteretes/cognee/tags/)
[![Downloads](https://static.pepy.tech/badge/cognee)](https://pepy.tech/project/cognee)



We build for developers who need a reliable, production-ready data layer for AI applications


## What is cognee?

Cognee implements scalable, modular ECL (Extract, Cognify, Load) pipelines that allow you to interconnect and retrieve past conversations, documents, and audio transcriptions while reducing hallucinations, developer effort, and cost.
Try it in a Google Colab  <a href="https://colab.research.google.com/drive/1g-Qnx6l_ecHZi0IOw23rg0qC4TYvEvWZ?usp=sharing">notebook</a>  or have a look at our <a href="https://docs.cognee.ai">documentation</a>

If you have questions, join our  <a href="https://discord.gg/NQPKmU5CCg">Discord</a> community

<div align="center">
<img src="assets/cognee_benefits.png" alt="why cognee" width="80%" />
</div>

## 📦 Installation

You can install Cognee using either **pip** or **poetry**.
Support for various databases and vector stores is available through extras.

### With pip

```bash
pip install cognee
```

### With poetry

```bash
poetry add cognee
```

### With pip with specific database support

To install Cognee with support for specific databases use the appropriate command below. Replace \<database> with the name of the database you need.
```bash
pip install 'cognee[<database>]'
```

Replace \<database> with any of the following databases:
- postgres
- weaviate
- qdrant
- neo4j
- milvus

Installing Cognee with PostgreSQL and Neo4j support example:
```bash
pip install 'cognee[postgres, neo4j]'
```

### With poetry with specific database support

To install Cognee with support for specific databases use the appropriate command below. Replace \<database> with the name of the database you need.
```bash
poetry add cognee -E <database>
```
Replace \<database> with any of the following databases:
- postgres
- weaviate
- qdrant
- neo4j
- milvus

Installing Cognee with PostgreSQL and Neo4j support example:
```bash
poetry add cognee -E postgres -E neo4j
```

## 💻 Basic Usage

### Setup

```
import os

os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"

```
or 
```
import cognee
cognee.config.set_llm_api_key("YOUR_OPENAI_API_KEY")
```
You can also set the variables by creating .env file, here is our <a href="https://github.com/topoteretes/cognee/blob/main/.env.template">template.</a>
To use different LLM providers, for more info check out our <a href="https://docs.cognee.ai">documentation</a>

If you are using Network, create an account on Graphistry to visualize results:
```
cognee.config.set_graphistry_config({
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD"
})
```

(Optional) To run the UI, go to cognee-frontend directory and run:
```
npm run dev
```
or run everything in a docker container:
```
docker-compose up
```
Then navigate to localhost:3000

If you want to use Cognee with PostgreSQL, make sure to set the following values in the .env file:
```
DB_PROVIDER=postgres

DB_HOST=postgres
DB_PORT=5432

DB_NAME=cognee_db
DB_USERNAME=cognee
DB_PASSWORD=cognee
``` 

### Simple example

First, copy `.env.template` to `.env` and add your OpenAI API key to the LLM_API_KEY field.

This script will run the default pipeline:

```python
import cognee
import asyncio
from cognee.api.v1.search import SearchType

async def main():
    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # cognee knowledge graph will be created based on this text
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """
    
    print("Adding text to cognee:")
    print(text.strip())  
    # Add the text, and make it available for cognify
    await cognee.add(text)
    print("Text added successfully.\n")

    
    print("Running cognify to create knowledge graph...\n")
    print("Cognify process steps:")
    print("1. Classifying the document: Determining the type and category of the input text.")
    print("2. Checking permissions: Ensuring the user has the necessary rights to process the text.")
    print("3. Extracting text chunks: Breaking down the text into sentences or phrases for analysis.")
    print("4. Adding data points: Storing the extracted chunks for processing.")
    print("5. Generating knowledge graph: Extracting entities and relationships to form a knowledge graph.")
    print("6. Summarizing text: Creating concise summaries of the content for quick insights.\n")
    
    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify()
    print("Cognify process complete.\n")

    
    query_text = 'Tell me about NLP'
    print(f"Searching cognee for insights with query: '{query_text}'")
    # Query cognee for insights on the added text
    search_results = await cognee.search(
        SearchType.INSIGHTS, query_text=query_text
    )
    
    print("Search results:")
    # Display results
    for result_text in search_results:
        print(result_text)

    # Example output:
       # ({'id': UUID('bc338a39-64d6-549a-acec-da60846dd90d'), 'updated_at': datetime.datetime(2024, 11, 21, 12, 23, 1, 211808, tzinfo=datetime.timezone.utc), 'name': 'natural language processing', 'description': 'An interdisciplinary subfield of computer science and information retrieval.'}, {'relationship_name': 'is_a_subfield_of', 'source_node_id': UUID('bc338a39-64d6-549a-acec-da60846dd90d'), 'target_node_id': UUID('6218dbab-eb6a-5759-a864-b3419755ffe0'), 'updated_at': datetime.datetime(2024, 11, 21, 12, 23, 15, 473137, tzinfo=datetime.timezone.utc)}, {'id': UUID('6218dbab-eb6a-5759-a864-b3419755ffe0'), 'updated_at': datetime.datetime(2024, 11, 21, 12, 23, 1, 211808, tzinfo=datetime.timezone.utc), 'name': 'computer science', 'description': 'The study of computation and information processing.'})
       # (...)
        #
        # It represents nodes and relationships in the knowledge graph:
        # - The first element is the source node (e.g., 'natural language processing').
        # - The second element is the relationship between nodes (e.g., 'is_a_subfield_of').
        # - The third element is the target node (e.g., 'computer science').

if __name__ == '__main__':
    asyncio.run(main())

```
When you run this script, you will see step-by-step messages in the console that help you trace the execution flow and understand what the script is doing at each stage.
A version of this example is here: `examples/python/simple_example.py`

### Understand our architecture

cognee framework consists of tasks that can be grouped into pipelines.
Each task can be an independent part of business logic, that can be tied to other tasks to form a pipeline.
These tasks persist data into your memory store enabling you to search for relevant context of past conversations, documents, or any other data you have stored.
<div align="center">
<img src="assets/cognee_diagram.png" alt="cognee concept diagram" width="50%" />
</div>


## Vector retrieval, Graphs and LLMs

Cognee supports a variety of tools and services for different operations:
- **Modular**: Cognee is modular by nature, using tasks grouped into pipelines

- **Local Setup**: By default, LanceDB runs locally with NetworkX and OpenAI.

- **Vector Stores**: Cognee supports LanceDB, Qdrant, PGVector and Weaviate for vector storage.

- **Language Models (LLMs)**: You can use either Anyscale or Ollama as your LLM provider.

- **Graph Stores**: In addition to NetworkX, Neo4j is also supported for graph storage.
  
- **User management**: Create individual user graphs and manage permissions

## Demo

Check out our demo notebook [here](https://github.com/topoteretes/cognee/blob/main/notebooks/cognee_demo.ipynb)


[<img src="https://i3.ytimg.com/vi/-ARUfIzhzC4/maxresdefault.jpg" width="100%">](https://www.youtube.com/watch?v=BDFt4xVPmro "Learn about cognee: 55")


## Get Started

### Install Server

Please see the [cognee Quick Start Guide](https://docs.cognee.ai/quickstart/) for important configuration information.

```bash
docker compose up
```


### Install SDK

Please see the cognee [Development Guide](https://docs.cognee.ai/quickstart/) for important beta information and usage instructions.

```bash
pip install cognee
```

## 💫 Contributors

<a href="https://github.com/topoteretes/cognee/graphs/contributors">
  <img alt="contributors" src="https://contrib.rocks/image?repo=topoteretes/cognee"/>
</a>


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=topoteretes/cognee&type=Date)](https://star-history.com/#topoteretes/cognee&Date)


## Vector & Graph Databases Implementation State



| Name     | Type               | Current state     | Known Issues |
|----------|--------------------|-------------------|--------------|
| Qdrant   | Vector             | Stable &#x2705;   |              |
| Weaviate | Vector             | Stable &#x2705;   |              |
| LanceDB  | Vector             | Stable &#x2705;   |              |
| Neo4j    | Graph              | Stable &#x2705;   |              |
| NetworkX | Graph              | Stable &#x2705;   |              |
| FalkorDB | Vector/Graph       | Unstable &#x274C; |              |
| PGVector | Vector             | Stable &#x2705;   |              |
| Milvus   | Vector             | Stable &#x2705;   |              |

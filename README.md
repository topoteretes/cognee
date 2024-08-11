# cognee

We build for developers who need a reliable, production-ready data layer for AI applications
cognee implements scalable, modular data pipelines that allow for creating the LLM-enriched data layer using graph and vector stores.



<p>
  <a href="https://cognee.ai" target="_blank">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/main/assets/cognee-logo.png" width="160px" alt="Cognee logo" />
  </a>
</p>


<p>
  <i> cognee aims to be dbt for LLMOps</i>
</p>

<p>
  <a href="https://github.com/topoteretes/cognee/fork">
    <img src="https://img.shields.io/github/forks/topoteretes/cognee?style=for-the-badge" alt="cognee forks"/>
  </a>
  <a href="https://github.com/topoteretes/cognee/stargazers">
    <img src="https://img.shields.io/github/stars/topoteretes/cognee?style=for-the-badge" alt="cognee stars"/>
  </a>
  <a href="https://github.com/topoteretes/cognee/pulls">
    <img src="https://img.shields.io/github/issues-pr/topoteretes/cognee?style=for-the-badge" alt="cognee pull-requests"/>
  </a>
  <a href="https://github.com/topoteretes/cognee/releases">
    <img src="https://img.shields.io/github/release/topoteretes/cognee?&label=Latest&style=for-the-badge" alt="cognee releases" />
  </a>
</p>



Try it in a Google collab  <a href="https://colab.research.google.com/drive/1jayZ5JRwDaUGFvCw9UZySBG-iB9gpYfu?usp=sharing">notebook</a>  or have a look at our <a href="https://topoteretes.github.io/cognee">documentation</a>




If you have questions, join our  <a href="https://discord.gg/NQPKmU5CCg">Discord</a> community





## 📦 Installation

### With pip

```bash
pip install cognee
```


### With poetry

```bash
poetry add cognee
```


## 💻 Usage

### Setup

```
import os

os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"

```
or 
```
import cognee
cognee.config.llm_api_key = "YOUR_OPENAI_API_KEY"
```
You can use different LLM providers, for more info check out our <a href="https://topoteretes.github.io/cognee">documentation</a>

In the next step make sure to launch a Postgres instance. Here is an example from our docker-compose:
```
  postgres:
    image: postgres:latest
    container_name: postgres
    environment:
      POSTGRES_USER: cognee
      POSTGRES_PASSWORD: cognee
      POSTGRES_DB: cognee_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - 5432:5432
    networks:
      - cognee-network
```


If you are using Networkx, create an account on Graphistry to visualize results:
```
   
   cognee.config.set_graphistry_username = "YOUR_USERNAME"
   cognee.config.set_graphistry_password = "YOUR_PASSWORD"
```

(Optional) To run the UI, run:
```
docker-compose up cognee
```
Then navigate to localhost:3000/wizard

### Run the default example

Make sure to launch the Postgres instance first. Navigate to the cognee folder and run:
```
docker compose up postgres
```

Run the default cognee pipeline:

```
import cognee

text = """Natural language processing (NLP) is an interdisciplinary
       subfield of computer science and information retrieval"""

await cognee.add([text], "example_dataset") # Add a new piece of information

await cognee.cognify() # Use LLMs and cognee to create knowledge

await search_results = cognee.search("SIMILARITY", {'query': 'Tell me about NLP'}) # Query cognee for the knowledge

print(search_results)

```


### Create your pipelines

cognee framework consists of tasks that can be grouped into pipelines. Each task can be an independent part of business logic, that can be tied to other tasks to form a pipeline.
Here is an example of how it looks for a default cognify pipeline:


1. To prepare the data for the pipeline run, first we need to add it to our metastore and normalize it:

Start with: 
```
docker compose up postgres
```
And then run: 
```
text = """Natural language processing (NLP) is an interdisciplinary
       subfield of computer science and information retrieval"""

await cognee.add([text], "example_dataset") # Add a new piece of information
```

2. In the next step we make a task. The task can be any business logic we need, but the important part is that it should be encapsulated in one function.

Here we show an example of creating a naive LLM classifier that takes a Pydantic model and then stores the data in both the graph and vector stores after analyzing each chunk.
We provided just a snippet for reference, but feel free to check out the implementation in our repo. 

```
async def chunk_naive_llm_classifier(data_chunks: list[DocumentChunk], classification_model: Type[BaseModel]):
    if len(data_chunks) == 0:
        return data_chunks

    chunk_classifications = await asyncio.gather(
        *[extract_categories(chunk.text, classification_model) for chunk in data_chunks],
    )

    classification_data_points = []

    for chunk_index, chunk in enumerate(data_chunks):
        chunk_classification = chunk_classifications[chunk_index]
        classification_data_points.append(uuid5(NAMESPACE_OID, chunk_classification.label.type))
        classification_data_points.append(uuid5(NAMESPACE_OID, chunk_classification.label.type))

        for classification_subclass in chunk_classification.label.subclass:
            classification_data_points.append(uuid5(NAMESPACE_OID, classification_subclass.value))

    vector_engine = get_vector_engine()

    class Keyword(BaseModel):
        uuid: str
        text: str
        chunk_id: str
        document_id: str

    collection_name = "classification"

    if await vector_engine.has_collection(collection_name):
        existing_data_points = await vector_engine.retrieve(
            collection_name,
            list(set(classification_data_points)),
        ) if len(classification_data_points) > 0 else []

        existing_points_map = {point.id: True for point in existing_data_points}
    else:
        existing_points_map = {}
        await vector_engine.create_collection(collection_name, payload_schema=Keyword)

    data_points = []
    nodes = []
    edges = []

    for (chunk_index, data_chunk) in enumerate(data_chunks):
        chunk_classification = chunk_classifications[chunk_index]
        classification_type_label = chunk_classification.label.type
        classification_type_id = uuid5(NAMESPACE_OID, classification_type_label)

...

```

To see existing tasks, have a look at the cognee.tasks


3. Once we have our tasks, it is time to group them into a pipeline.
This snippet shows how a group of tasks can be added to a pipeline, and how they can pass the information forward from one to another. 

```
            tasks = [
                Task(document_to_ontology, root_node_id = root_node_id),
                Task(source_documents_to_chunks, parent_node_id = root_node_id), # Classify documents and save them as a nodes in graph db, extract text chunks based on the document type
                Task(chunk_to_graph_decomposition, topology_model = KnowledgeGraph, task_config = { "batch_size": 10 }), # Set the graph topology for the document chunk data
                Task(chunks_into_graph, graph_model = KnowledgeGraph, collection_name = "entities"), # Generate knowledge graphs from the document chunks and attach it to chunk nodes
                Task(chunk_update_check, collection_name = "chunks"), # Find all affected chunks, so we don't process unchanged chunks
                Task(
                    save_chunks_to_store,
                    collection_name = "chunks",
                ), # Save the document chunks in vector db and as nodes in graph db (connected to the document node and between each other)
                run_tasks_parallel([
                    Task(
                        chunk_extract_summary,
                        summarization_model = cognee_config.summarization_model,
                        collection_name = "chunk_summaries",
                    ), # Summarize the document chunks
                    Task(
                        chunk_naive_llm_classifier,
                        classification_model = cognee_config.classification_model,
                    ),
                ]),
                Task(chunk_remove_disconnected), # Remove the obsolete document chunks.
            ]

            pipeline = run_tasks(tasks, documents)

```

To see the working code, check cognee.api.v1.cognify default pipeline in our repo.


## Vector retrieval, Graphs and LLMs

Cognee supports a variety of tools and services for different operations:
- **Modular**: Cognee is modular by nature, using tasks grouped into pipelines

- **Local Setup**: By default, LanceDB runs locally with NetworkX and OpenAI.

- **Vector Stores**: Cognee supports Qdrant and Weaviate for vector storage.

- **Language Models (LLMs)**: You can use either Anyscale or Ollama as your LLM provider.

- **Graph Stores**: In addition to LanceDB, Neo4j is also supported for graph storage.
  
- **User management**: Create individual user graphs and manage permissions

## Demo

Check out our demo notebook [here](https://github.com/topoteretes/cognee/blob/main/notebooks/cognee%20-%20Get%20Started.ipynb)



[<img src="https://i3.ytimg.com/vi/-ARUfIzhzC4/maxresdefault.jpg" width="100%">](https://www.youtube.com/watch?v=BDFt4xVPmro "Learn about cognee: 55")



## Star History


[![Star History Chart](https://api.star-history.com/svg?repos=topoteretes/cognee&type=Date)](https://star-history.com/#topoteretes/cognee&Date)

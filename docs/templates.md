# TASKS

Cognee uses tasks grouped into pipelines that populate graph and vector stores. These tasks analyze and enrich data, enhancing the quality of answers produced by Large Language Models (LLMs).

This section provides a template to help you structure your data and build pipelines. \
These tasks serve as a starting point for using Cognee to create reliable LLM pipelines.


## Task 1: Category Extraction

Data enrichment is the process of enhancing raw data with additional information to make it more valuable. This template is a sample task that extracts categories from a document and populates a graph with the extracted categories.

Let's go over the steps to use this template [full code provided here](https://github.com/topoteretes/cognee/blob/main/cognee/tasks/chunk_naive_llm_classifier/chunk_naive_llm_classifier.py):


This function is designed to classify chunks of text using a specified language model. The goal is to categorize the text, map relationships, and store the results in a vector engine and a graph engine. The function is asynchronous, allowing for concurrent execution of tasks like classification and data point creation.

### Parameters

- `data_chunks: list[DocumentChunk]`: A list of text chunks to be classified. Each chunk represents a piece of text and includes metadata like `chunk_id` and `document_id`.
- `classification_model: Type[BaseModel]`: The model used to classify each chunk of text. This model is expected to output labels that categorize the text.

### Steps in the Function

#### Check for Empty Input

```python
if len(data_chunks) == 0:
    return data_chunks
```

If there are no data chunks provided, the function returns immediately with the input list (which is empty).

#### Classify Each Chunk

```python
chunk_classifications = await asyncio.gather(
    *[extract_categories(chunk.text, classification_model) for chunk in data_chunks],
)
```

The function uses `asyncio.gather` to concurrently classify each chunk of text. `extract_categories` is called for each chunk, and the results are collected in `chunk_classifications`.

#### Initialize Data Structures

```python
classification_data_points = []
```

A list is initialized to store the classification data points that will be used later for mapping relationships and storing in the vector engine.

#### Generate UUIDs for Classifications

The function loops through each chunk and generates unique identifiers (UUIDs) for both the main classification type and its subclasses:

```python
classification_data_points.append(uuid5(NAMESPACE_OID, chunk_classification.label.type))
classification_data_points.append(uuid5(NAMESPACE_OID, classification_subclass.value))
```

These UUIDs are used to uniquely identify classifications and ensure consistency.

#### Retrieve or Create Vector Collection

```python
vector_engine = get_vector_engine()
collection_name = "classification"
```

The function interacts with a vector engine. It checks if the collection named "classification" exists. If it does, it retrieves existing data points to avoid duplicates. Otherwise, it creates the collection.

#### Prepare Data Points, Nodes, and Edges

The function then builds a list of `data_points` (representing the classification results) and constructs nodes and edges to represent relationships between chunks and their classifications:

```python
data_points.append(DataPoint[Keyword](...))
nodes.append((...))
edges.append((...))
```

- **Nodes**: Represent classifications (e.g., media type, subtype).
- **Edges**: Represent relationships between chunks and classifications (e.g., "is_media_type", "is_subtype_of").

#### Create Data Points and Relationships

If there are new nodes or edges to add, the function stores the data points in the vector engine and updates the graph engine with the new nodes and edges:

```python
await vector_engine.create_data_points(collection_name, data_points)
await graph_engine.add_nodes(nodes)
await graph_engine.add_edges(edges)
```

#### Return the Processed Chunks

Finally, the function returns the processed `data_chunks`, which can now be used further as needed:

```python
return data_chunks
```

## Pipeline 1: cognee.cognify (main pipeline)

This is the main pipeline currently implemented in cognee. It is designed to process data in a structured way and populate the graph and vector stores with the results.


This function is the entry point for processing datasets. It handles dataset retrieval, user authorization, and manages the execution of a pipeline of tasks that process documents.

### Parameters

- `dataset: Union[str, list[str]] = None`: A string or list of dataset names to be processed.
- `user: User = None`: The user requesting the processing. If not provided, the default user is retrieved.

### Steps in the Function

#### User Authentication

```python
if user is None:
    user = await get_default_user()
```

If no user is provided, the function retrieves the default user.

#### Handling Empty or String Dataset Input

```python
existing_datasets = await get_datasets(user.id)
if datasets is None or len(datasets) == 0:
        datasets = existing_datasets
if type(datasets[0]) == str:
        datasets = await get_datasets_by_name(datasets, user.id)
```

If no datasets are provided, the function retrieves all datasets owned by the user. If a list of dataset names (strings) is provided, they are converted into dataset objects.

#### Run Cognify Pipeline for Each Dataset

```python
existing_datasets_map = {
        generate_dataset_name(dataset.name): True for dataset in existing_datasets
    }

awaitables = []

for dataset in datasets:
    dataset_name = generate_dataset_name(dataset.name)

    if dataset_name in existing_datasets_map:
        awaitables.append(run_cognify_pipeline(dataset, user))

    return await asyncio.gather(*awaitables)
```

The `run_cognify_pipeline` function is defined within `cognify` and is responsible for processing a single dataset. This is where most of the heavy lifting occurs.

#### Pipeline Tasks

The pipeline consists of several tasks, each responsible for different parts of the processing:

- `classify_documents`: Converts each of the documents into one of the specific Document types: PdfDocument, AudioDocument, ImageDocument or TextDocument
- `check_permissions_on_documents`: Checks if the user has the necessary permissions to access the documents. In this case, it checks for "write" permission.
- `extract_chunks_from_documents`: Extracts text chunks based on the document type.
- `add_data_points`:
- `extract_graph_from_data`: Generates knowledge graphs from the document chunks.
- `summarize_text`: Extracts a summary for each chunk using an llm.

The tasks are managed and executed asynchronously using the `run_tasks` and `run_tasks_parallel` functions.

```python
pipeline = run_tasks(tasks, documents)
async for result in pipeline:
    print(result)
```
#### Processing Multiple Datasets

The function prepares to process multiple datasets concurrently using `asyncio.gather`.

```python
awaitables = []
for dataset in datasets:
    dataset_name = generate_dataset_name(dataset.name)
    if dataset_name in existing_datasets:
        awaitables.append(run_cognify_pipeline(dataset))
return await asyncio.gather(*awaitables)
```

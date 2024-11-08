# PIPELINES

Cognee uses [tasks](https://github.com/topoteretes/cognee/blob/main/cognee/modules/pipelines/tasks/Task.py) grouped into pipelines that populate graph and vector stores. [These tasks](https://github.com/topoteretes/cognee/tree/main/cognee/tasks) analyze and enrich data, enhancing the quality of answers produced by Large Language Models (LLMs). 

The tasks are managed and executed asynchronously using the `run_tasks` and `run_tasks_parallel` functions.

```python
pipeline = run_tasks(tasks, documents)
async for result in pipeline:
    print(result)
```

## Main pipeline: [cognee.cognify](https://github.com/topoteretes/cognee/blob/168cb5d1bf1964b5b0c645b2f3d8638d84554fda/cognee/api/v1/cognify/cognify_v2.py#L38)

This is the main pipeline currently implemented in cognee. It is designed to process data in a structured way and populate the graph and vector stores.


This function is the entry point for processing datasets. It handles dataset retrieval, user authorization, and manages the execution of a pipeline of tasks that process documents.

### Parameters

- `datasets: Union[str, list[str]] = None`: A string or list of dataset names to be processed.
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

#### Selecting datasets from the input list that are owned by the user

```python
existing_datasets_map = {
        generate_dataset_name(dataset.name): True for dataset in existing_datasets
    }
```

#### Run Cognify Pipeline for Each Dataset

```python
awaitables = []

for dataset in datasets:
    dataset_name = generate_dataset_name(dataset.name)

    if dataset_name in existing_datasets_map:
        awaitables.append(run_cognify_pipeline(dataset, user))

return await asyncio.gather(*awaitables)

The `run_cognify_pipeline` function is defined within `cognify` and is responsible for processing a single dataset. This is where most of the heavy lifting occurs. The function processes multiple datasets concurrently using `asyncio.gather`.


#### Pipeline Tasks

The pipeline consists of several tasks, each responsible for different parts of the processing:

- `classify_documents`: Converts each of the documents into one of the specific Document types: PdfDocument, AudioDocument, ImageDocument or TextDocument
- `check_permissions_on_documents`: Checks if the user has the necessary permissions to access the documents. In this case, it checks for "write" permission.
- `extract_chunks_from_documents`: Extracts text chunks based on the document type.
- `add_data_points`: Creates nodes and edges from the chunks and their properties. Adds them to the graph engine.
- `extract_graph_from_data`: Generates knowledge graphs from the document chunks.
- `summarize_text`: Extracts a summary for each chunk using an llm.

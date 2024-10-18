# Cognee API Reference

## Overview

The Cognee API provides a set of endpoints for managing datasets, performing cognitive tasks, and configuring various settings in the system. The API is built on FastAPI and includes multiple routes to handle different functionalities. This reference outlines the available endpoints and their usage.

## Base URL

The base URL for all API requests is determined by the server's deployment environment. Typically, this will be:

- **Development**: `http://localhost:8000`
- **Production**: Depending on your server setup.

## Endpoints

### 1. Root

- **URL**: `/`
- **Method**: `GET`
- **Auth Required**: No
- **Description**: Root endpoint that returns a welcome message.
  
  **Response**:
  ```json
  {
    "status": 200,
    "body": {
      "message": "Hello, World, I am alive!"
    }
  }
  ```

### 2. Health Check

- **URL**: `/health`
- **Method**: `GET`
- **Auth Required**: No
- **Description**: Health check endpoint that returns the server status.
  
  **Response**:
  ```json
  {
    "status": 200
  }
  ```

### 3. Get Datasets

- **URL**: `/datasets`
- **Method**: `GET`
- **Auth Required**: No
- **Description**: Retrieve a list of available datasets.
  
  **Response**:
  ```json
  {
    "status": 200,
    "body": [
      {
        "id": "dataset_id_1",
        "name": "Dataset Name 1",
        "description": "Description of Dataset 1",
        ...
      },
      ...
    ]
  }
  ```

### 4. Delete Dataset

- **URL**: `/datasets/{dataset_id}`
- **Method**: `DELETE`
- **Auth Required**: No
- **Description**: Delete a specific dataset by its ID.
  
  **Path Parameters**:
  - `dataset_id`: The ID of the dataset to delete.
  
  **Response**:
  ```json
  {
    "status": 200
  }
  ```

### 5. Get Dataset Graph

- **URL**: `/datasets/{dataset_id}/graph`
- **Method**: `GET`
- **Auth Required**: No
- **Description**: Retrieve the graph visualization URL for a specific dataset.
  
  **Path Parameters**:
  - `dataset_id`: The ID of the dataset.
  
  **Response**:
  ```json
  "http://example.com/path/to/graph"
  ```

### 6. Get Dataset Data

- **URL**: `/datasets/{dataset_id}/data`
- **Method**: `GET`
- **Auth Required**: No
- **Description**: Retrieve data associated with a specific dataset.
  
  **Path Parameters**:
  - `dataset_id`: The ID of the dataset.
  
  **Response**:
  ```json
  {
    "status": 200,
    "body": [
      {
        "data_id": "data_id_1",
        "content": "Data content here",
        ...
      },
      ...
    ]
  }
  ```

### 7. Get Dataset Status

- **URL**: `/datasets/status`
- **Method**: `GET`
- **Auth Required**: No
- **Description**: Retrieve the status of one or more datasets.
  
  **Query Parameters**:
  - `dataset`: A list of dataset IDs to check status for.
  
  **Response**:
  ```json
  {
    "status": 200,
    "body": {
      "dataset_id_1": "Status 1",
      "dataset_id_2": "Status 2",
      ...
    }
  }
  ```

### 8. Get Raw Data

- **URL**: `/datasets/{dataset_id}/data/{data_id}/raw`
- **Method**: `GET`
- **Auth Required**: No
- **Description**: Retrieve the raw data file for a specific data entry in a dataset.
  
  **Path Parameters**:
  - `dataset_id`: The ID of the dataset.
  - `data_id`: The ID of the data entry.
  
  **Response**: Raw file download.

### 9. Add Data

- **URL**: `/add`
- **Method**: `POST`
- **Auth Required**: No
- **Description**: Add new data to a dataset. The data can be uploaded from a file or a URL.
  
  **Form Parameters**:
  - `datasetId`: The ID of the dataset to add data to.
  - `data`: A list of files to upload.

  **Request**
  ```json
  {
    "dataset_id": "ID_OF_THE_DATASET_TO_PUT_DATA_IN", // Optional, we use "main" as default.
    "files": File[]
  }
  ```
  
  **Response**:
  ```json
  {
    "status": 200
  }
  ```

### 10. Cognify

- **URL**: `/cognify`
- **Method**: `POST`
- **Auth Required**: No
- **Description**: Perform cognitive processing on the specified datasets.
  
  **Request Body**:
  ```json
  {
    "datasets": ["ID_OF_THE_DATASET_1", "ID_OF_THE_DATASET_2", ...]
  }
  ```
  
  **Response**:
  ```json
  {
    "status": 200
  }
  ```

### 11. Search

- **URL**: `/search`
- **Method**: `POST`
- **Auth Required**: No
- **Description**: Search for nodes in the graph based on the provided query parameters.
  
  **Request Body**:
  ```json
  {
    "searchType": "INSIGHTS", // Or "SUMMARIES" or "CHUNKS"
    "query": "QUERY_TO_MATCH_DATA"
  }
  ```

  **Response**

  For "INSIGHTS" search type:
  ```json
  {
    "status": 200,
    "body": [[
      { "name" "source_node_name" },
      { "relationship_name" "between_nodes_relationship_name" },
      { "name" "target_node_name" },
    ]]
  }
  ```

  For "SUMMARIES" search type:
    ```json
    {
      "status": 200,
      "body": [
        { "text" "summary_text" },
        { "text" "summary_text" },
        { "text" "summary_text" },
        ...
      ]
    }
    ```

  For "CHUNKS" search type:
  ```json
  {
    "status": 200,
    "body": [
      { "text" "chunk_text" },
      { "text" "chunk_text" },
      { "text" "chunk_text" },
      ...
    ]
  }
  ```

### 12. Get Settings

- **URL**: `/settings`
- **Method**: `GET`
- **Auth Required**: No
- **Description**: Retrieve the current system settings.
  
  **Response**:
  ```json
  {
    "status": 200,
    "body": {
      "llm": {...},
      "vectorDB": {...},
      ...
    }
  }
  ```

### 13. Save Settings

- **URL**: `/settings`
- **Method**: `POST`
- **Auth Required**: No
- **Description**: Save new settings for the system, including LLM and vector DB configurations.
  
  **Request Body**:
  - `llm`: Optional. The configuration for the LLM provider.
  - `vectorDB`: Optional. The configuration for the vector database provider.
  
  **Response**:
  ```json
  {
    "status": 200
  }
  ```

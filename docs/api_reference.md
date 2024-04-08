# cognee API Reference

## Overview
The Cognee API is a FastAPI server that provides endpoints for adding data to a graph, cognitive processing of content, and searching for nodes in the graph.

## Endpoints
For each API endpoint, provide the following details:



### Endpoint 1: Root
- URL: /add
- Method: POST
- Auth Required: No
- Description: Root endpoint that returns a welcome message.

#### Response
```json
{
  "message": "Hello, World, I am alive!"
}
```

### Endpoint 1: Health Check
- URL: /health
- Method: GET
- Auth Required: No
- Description: Health check endpoint that returns the server status.
#### Response
```json
{
  "status": "OK"
}
```
### Endpoint 2: Add
- URL: /Add
- Method: POST 
- Auth Required: Yes | No
- Description: This endpoint is responsible for adding data to the graph.

#### Parameters
| Name | Type                                             | Required | Description |
| --- |--------------------------------------------------| --- | --- |
| data | Union[str, BinaryIO, List[Union[str, BinaryIO]]] | Yes | The data to be added|
| dataset_id | UUID                                             | Yes | The ID of the dataset. |
| dataset_name | String                                           | Yes | The name of the dataset.|



#### Response
```json
{
  "response": "data"
}
```

### Endpoint 3: Cognify
- URL: /cognify
- Method: POST 
- Auth Required: Yes | No
- Description: This endpoint is responsible for the cognitive processing of the content.

#### Parameters
| Name | Type                                             | Required | Description |
| --- |--------------------------------------------------| --- | --- |
| datasets | Union[str, List[str]] | Yes | The data to be added|


#### Response
```json
{
  "response": "data"
}
```


### Endpoint 4: search
- URL: /search
- Method: POST 
- Auth Required: No
- Description: This endpoint is responsible for searching for nodes in the graph.
#### Parameters
| Name | Type | Required | Description |
| --- | --- | --- | --- |
| query_params | Dict[str, Any] | Yes | Description of the parameter. |


#### Response
```json
{
  "response": "data"
}
```
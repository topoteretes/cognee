## PromethAI Memory Manager



### Description


RAG test manager can be used via API (in progress) or via the CLI

Make sure to run scripts/create_database.py

After that, you can run: 

``` python test_runner.py \
    --url "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf" \
    --test_set "path/to/test_set.json" \
    --user_id "666" \
    --metadata "path/to/metadata.json" 
```


#How to start 

## Installation

```docker compose build promethai_mem   ```

## Run the level 3

Make sure you have Docker, Poetry, and Python 3.11 installed and postgres installed.

Copy the .env.example to .env and fill the variables


Start the docker:

```docker compose up promethai_mem   ```

Use the poetry environment:

``` poetry shell ```

Make sure to run to initialize DB tables

``` python scripts/create_database.py ```

After that, you can run the RAG test manager.


``` 
    python rag_test_manager.py \
    --url "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf" \
    --test_set "example_data/test_set.json" \
    --user_id "666" \
    --metadata "example_data/metadata.json"

```
Examples of metadata structure and test set are in the folder "example_data"

To analyze your data, go to your local Superset instance:

``` 
    http://localhost:8088
```

Add the  Postgres datasource to the Superset with the following connection string:
    
``` 
    postgres://bla:bla@postgres:5432/bubu
```


## Clean database

```docker compose down promethai_mem   ```


```docker volume prune  ```

``` docker compose up --force-recreate --build promethai_mem ```


## Usage

The fast API endpoint accepts prompts and stores data with the help of the Memory Manager

The types of memory are: Episodic, Semantic, Buffer

Endpoint Overview
The Memory API provides the following endpoints:

- /[memory_type]/add-memory (POST)
- /[memory_type]/fetch-memory (POST)
- /[memory_type]/delete-memory (POST)
- /available-buffer-actions (GET)
- /run-buffer (POST)
- /buffer/create-context (POST)



## How To Get Started

1. We do a post request to add-memory endpoint with the following payload:
It will upload Jack London "Call of the Wild" to SEMANTIC memory
```
curl -X POST http://localhost:8000/semantic/add-memory -H "Content-Type: application/json" -d '{
  "payload": {
    "user_id": "681",
    "prompt": "I am adding docs",
    "params": {
        "version": "1.0",
        "agreement_id": "AG123456",
        "privacy_policy": "https://example.com/privacy",
        "terms_of_service": "https://example.com/terms",
        "format": "json",
        "schema_version": "1.1",
        "checksum": "a1b2c3d4e5f6",
        "owner": "John Doe",
        "license": "MIT",
        "validity_start": "2023-08-01",
        "validity_end": "2024-07-31"
    },
    "loader_settings": {
        "format": "PDF",
        "source": "url",
        "path": "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"
    }
  }
}'
```

2. We run the buffer with the prompt "I want to know how does Buck adapt to life in the wild and then have that info translated to German "

```
curl -X POST http://localhost:8000/run-buffer -H "Content-Type: application/json" -d '{
  "payload": {
    "user_id": "681",
    "prompt": "I want to know how does Buck adapt to life in the wild and then have that info translated to German ",
    "params": {
        "version": "1.0",
        "agreement_id": "AG123456",
        "privacy_policy": "https://example.com/privacy",
        "terms_of_service": "https://example.com/terms",
        "format": "json",
        "schema_version": "1.1",
        "checksum": "a1b2c3d4e5f6",
        "owner": "John Doe",
        "license": "MIT",
        "validity_start": "2023-08-01",
        "validity_end": "2024-07-31"
    },
    "attention_modulators": {
        "relevance": 0.0,
        "saliency": 0.1
    }
  }
}'
```


Other attention modulators that could be implemented: 

        "frequency": 0.5, 
        "repetition": 0.5,
        "length": 0.5,
        "position": 0.5,
        "context": 0.5,
        "emotion": 0.5,
        "sentiment": 0.5,
        "perspective": 0.5,
        "style": 0.5,
        "grammar": 0.5,
        "spelling": 0.5,
        "logic": 0.5,
        "coherence": 0.5,
        "cohesion": 0.5,
        "plausibility": 0.5,
        "consistency": 0.5,
        "informativeness": 0.5,
        "specificity": 0.5,
        "detail": 0.5,
        "accuracy": 0.5,
        "topicality": 0.5,
        "focus": 0.5,
        "clarity": 0.5,
        "simplicity": 0.5,
        "naturalness": 0.5,
        "fluency": 0.5,
        "variety": 0.5,
        "vividness": 0.5,
        "originality": 0.5,
        "creativity": 0.5,
        "humor": 0.5,
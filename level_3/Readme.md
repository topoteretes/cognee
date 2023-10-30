#### Docker: 

Copy the .env.template to .env and fill in the variables
Specify the environment variable in the .env file to "docker"


Launch the docker image:

```docker compose up promethai_mem  ```

Send the request to the API:

```
curl -X POST -H "Content-Type: application/json" -d '{
  "payload": {
    "user_id": "681",
    "data": [".data/3ZCCCW.pdf"],
    "test_set": "sample",
    "params": ["chunk_size"],
    "metadata": "sample",
    "retriever_type": "single_document_context"
  }
}' http://0.0.0.0:8000/rag-test/rag_test_run
 
```
Params:

- data -> list of URLs or path to the file, located in the .data folder (pdf, docx, txt, html)
- test_set -> sample, manual (list of questions and answers)
- metadata -> sample,  manual (json) or version (in progress)
- params -> chunk_size, chunk_overlap, search_type (hybrid, bm25), embeddings
- retriever_type -> llm_context, single_document_context, multi_document_context, cognitive_architecture(coming soon)

Inspect the results in the DB:

``` docker exec -it postgres psql -U bla ```

``` \c bubu ```

``` select * from test_outputs; ```

Or set up the superset to visualize the results:



#### Poetry environment: 


Copy the .env.template to .env and fill in the variables
Specify the environment variable in the .env file to "local"

Use the poetry environment:

``` poetry shell ```

Change the .env file Environment variable to "local"

Launch the postgres DB

``` docker compose up postgres ```

Launch the superset

``` docker compose up superset ```

Open the superset in your browser

``` http://localhost:8088 ```
Add the  Postgres datasource to the Superset with the following connection string:
    
``` postgres://bla:bla@postgres:5432/bubu ```

Make sure to run to initialize DB tables

``` python scripts/create_database.py ```

After that, you can run the RAG test manager from your command line.


``` 
    python rag_test_manager.py \
    --file ".data" \
    --test_set "example_data/test_set.json" \
    --user_id "666" \
    --params "chunk_size" "search_type" \
    --metadata "example_data/metadata.json" \
    --retriever_type "single_document_context"

```

Examples of metadata structure and test set are in the folder "example_data"

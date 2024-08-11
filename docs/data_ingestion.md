# How data ingestion with cognee works




# Why bother with data ingestion?

In order to use cognee, you need to ingest data into the cognee data store. 
This data can be events, customer data, or third-party data. 

In order to build reliable models and pipelines, we need to structure and process various types of datasets and data sources in the same way.
Some of the operations like normalization, deduplication, and data cleaning are common across all data sources.


This is where cognee comes in. It provides a unified interface to ingest data from various sources and process it in a consistent way.
For this we use dlt (Data Loading Tool) which is a part of cognee infrastructure.


# Example

Let's say you have a dataset of customer reviews in a PDF file. You want to ingest this data into cognee and use it to train a model.

You can use the following code to ingest the data:

```python
dataset_name = "artificial_intelligence"

ai_text_file_path = os.path.join(pathlib.Path(__file__).parent, "test_data/artificial-intelligence.pdf")
await cognee.add([ai_text_file_path], dataset_name)

```

cognee uses dlt to ingest the data and allows you to use:

1. SQL databases. Supports PostgreSQL, MySQL, MS SQL Server, BigQuery, Redshift, and more.
2. REST API generic source. Loads data from REST APIs using declarative configuration.
3. OpenAPI source generator. Generates a source from an OpenAPI 3.x spec using the REST API source.
4. Cloud and local storage. Retrieves data from AWS S3, Google Cloud Storage, Azure Blob Storage, local files, and more.



# What happens under the hood?

We use dlt as a loader to ingest data into the cognee metadata store. We can ingest data from various sources like SQL databases, REST APIs, OpenAPI specs, and cloud storage.
This enables us to have a common data model we can then use to build models and pipelines.
The models and pipelines we build in this way end up in the cognee data store, which is a unified interface to access the data.
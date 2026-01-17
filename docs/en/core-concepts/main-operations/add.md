# Add

> Ingesting and preparing data for processing in Cognee

## What is the add operation

The `.add` operation is how you bring content into Cognee. It takes your files, directories, or raw text, normalizes them into plain text, and records them into a dataset that Cognee can later expand into vectors and graphs with [Cognify](../main-operations/cognify).

* **Ingestion-only**: no embeddings, no graph yet
* **Flexible input**: raw text, local files, directories, or S3 URIs
* **Normalized storage**: everything is turned into text and stored consistently
* **Deduplicated**: Cognee uses content hashes to avoid duplicates
* **Dataset-first**: everything you add goes into a dataset
  * Datasets are how Cognee keeps different collections organized (e.g. "research-papers", "customer-reports")
  * Each dataset has its own ID, owner, and permissions for access control
  * You can read more about them below

## Where add fits

* First step before you run [Cognify](../main-operations/cognify)
* Use it to **create a dataset** from scratch, or **append new data** over time
* Ideal for both local experiments and programmatic ingestion from storage (e.g. S3)

## What happens under the hood

1. **Expand your input**
   * Directories are walked, S3 paths are expanded, raw text is passed through
   * Result: a flat list of items (files, text, handles)

2. **Ingest and register**
   * Files are saved into Cognee's storage and converted to text
   * Cognee computes a stable content hash to prevent duplicates
   * Each item becomes a record in the database and is attached to your dataset
   * **Text extraction**: Converts various file formats into plain text
   * **Metadata preservation**: Keeps file information like source, creation date, and format
   * **Content normalization**: Ensures consistent text encoding and formatting

3. **Return a summary**
   * You get a pipeline run info object that tells you where everything went and which dataset is ready for the next step

## After add finishes

After `.add` completes, your data is ready for the next stage:

* **Files are safely stored** in Cognee's storage system with metadata preserved
* **Database records** track each ingested item and link it to your dataset
* **Dataset is prepared** for transformation with [Cognify](../main-operations/cognify) — which will chunk, embed, and connect everything

## Further details

<Accordion title="Input sources">
  * Mix and match: `["some text", "/path/to/file.pdf", "s3://bucket/data.csv"]`
  * Works with directories (recursively), S3 prefixes, and file handles
  * Local and cloud sources are normalized into the same format
</Accordion>

<Accordion title="Supported formats">
  * **Text**: `.txt, .md, .csv, .json, …`
  * **PDF**: `.pdf`
  * **Images**: common formats like `.png, .jpg, .gif, .webp, …`
  * **Audio**: `.mp3, .wav, .flac, …`
  * **Office docs**: `.docx, .pptx, .xlsx, …`
  * **Docling**: Cognee can also ingest the `DoclingDocument` format. Any format that [Docling](https://github.com/docling-project/docling) supports as input can be converted, then passed on to Cognee's add.
  * Cognee chooses the right loader for each format under the hood
</Accordion>

<Accordion title="Datasets">
  * A dataset is your "knowledge base" — a grouping of related data that makes sense together
  * Datasets are **first-class objects in Cognee's database** with their own ID, name, owner, and permissions
  * They provide **scope**: `.add` writes into a dataset, [Cognify](../main-operations/cognify) processes per-dataset
  * Think of them as separate shelves in your library — e.g., a "research-papers" dataset and a "customer-reports" dataset
  * If you name a dataset that doesn't exist, Cognee creates it for you; if you don't specify, a default one is used
</Accordion>

<Accordion title="Users and ownership">
  * Every dataset and data item belongs to a user
  * If you don't pass a user, Cognee creates/uses a default one
  * Ownership controls who can later read, write, or share that dataset
</Accordion>

<Accordion title="Node sets">
  * Optional labels to group or tag data on ingestion
  * Example: `node_set=["AI", "FinTech"]`
  * Useful later when you want to focus on subgraphs
</Accordion>

<Columns cols={3}>
  <Card title="Cognify" icon="brain-cog" href="/core-concepts/main-operations/cognify">
    Expand data into chunks, embeddings, and graphs
  </Card>

  <Card title="DataPoints" icon="circle" href="/core-concepts/building-blocks/datapoints">
    The units you'll see after Cognify
  </Card>

  <Card title="Building Blocks" icon="puzzle" href="/core-concepts/building-blocks/tasks">
    Learn about Tasks and Pipelines behind Add
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
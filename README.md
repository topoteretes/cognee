# PromethAI-Memory
Memory management and testing for the AI Applications and RAGs



![Infographic Image](https://github.com/topoteretes/PromethAI-Memory/blob/main/infographic_final.png)

## Production-ready modern data platform

Browsing the database of theresanaiforthat.com, we can observe around [7000 new, mostly semi-finished projects](https://theresanaiforthat.com/) in the field of applied AI.
It seems it has never been easier to create a startup, build an app, and go to market… and fail.
Decades of technological advancements have led to small teams being able to do in 2023 what in 2015 required a team of dozens.
Yet, the AI apps currently being pushed out still mostly feel and perform like demos.
The rise of this new profession is perhaps signaling the need for a solution that is not yet there — a solution that in its essence represents a Large Language Model (LLM) — [a powerful general problem solver](https://lilianweng.github.io/posts/2023-06-23-agent/?fbclid=IwAR1p0W-Mg_4WtjOCeE8E6s7pJZlTDCDLmcXqHYVIrEVisz_D_S8LfN6Vv20) — available in the palm of your hand 24/7/365.

To address this issue, [dlthub](https://dlthub.com/) and [prometh.ai](http://prometh.ai/) will collaborate on a productionizing a common use-case, progressing step by step. We will utilize the LLMs, frameworks, and services, refining the code until we attain a clearer understanding of what a modern LLM architecture stack might entail.

## Read more on our blog post [prometh.ai](http://prometh.ai/promethai-memory-blog-post-on)


## Project Structure

### Level 1 - OpenAI functions + Pydantic + DLTHub
Scope: Give PDFs to the model and get the output in a structured format
We introduce the following concepts:
- Structured output with Pydantic
- CMD script to process custom PDFs
### Level 2 - Memory Manager + Metadata management
Scope: Give PDFs to the model and consolidate with the previous user activity and more
We introduce the following concepts:

- Long Term Memory -> store and format the data
- Episodic Buffer -> isolate the working memory
- Attention Modulators -> improve semantic search
- Docker
- API

### Level 3 - Dynamic Memory Manager + DB + Rag Test Manager
Scope: Store the data in N stores and test the retrieval with the Rag Test Manager
- Dynamic Memory Manager -> store the data in N stores
- Auto-generation of tests
- Multiple file formats supported
- Postgres DB to manage state
- Docker
- API


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

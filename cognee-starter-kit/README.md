
# Cognee Starter Kit
Welcome to the <a href="https://github.com/topoteretes/cognee">cognee</a> Starter Repo! This repository is designed to help you get started quickly by providing a structured dataset and pre-built data pipelines using cognee to build powerful knowledge graphs.

You can use this repo to ingest, process, and visualize data in minutes. 

By following this guide, you will:

- Load structured company and employee data
- Utilize pre-built pipelines for data processing
- Perform graph-based search and query operations
- Visualize entity relationships effortlessly on a graph

# How to Use This Repo 🛠

## Install uv if you don't have it on your system
```
pip install uv
```
## Install dependencies
```
uv sync
```

## Setup LLM
Add environment variables to `.env` file.
In case you choose to use OpenAI provider, add just the model and api_key.

**Supported LLM Providers:**
- OpenAI (default)
- Anthropic
- Gemini
- Ollama
- AWS Bedrock

```
LLM_PROVIDER=""
LLM_MODEL=""
LLM_ENDPOINT=""
LLM_API_KEY=""
LLM_API_VERSION=""

EMBEDDING_PROVIDER=""
EMBEDDING_MODEL=""
EMBEDDING_ENDPOINT=""
EMBEDDING_API_KEY=""
EMBEDDING_API_VERSION=""
```

**For AWS Bedrock, you have three authentication options:**

1. **API Key (Bearer Token):**
```
LLM_PROVIDER="bedrock"
LLM_API_KEY="your_bedrock_api_key"
LLM_MODEL="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
AWS_REGION_NAME="us-east-1"
```

2. **AWS Credentials:**
```
LLM_PROVIDER="bedrock"
LLM_MODEL="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
[if needed] AWS_SESSION_TOKEN="your_session_token"
AWS_REGION_NAME="us-east-1"
```

3. **AWS Profile:**
```
LLM_PROVIDER="bedrock"
LLM_MODEL="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
AWS_PROFILE_NAME="your_aws_profile"
AWS_REGION_NAME="us-east-1"
```

**Note:** For Bedrock models, use an [inference profile](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html#API_runtime_InvokeModel_Example_5:~:text=Use%20an%20inference%20profile%20in%20model%20invocation) for `LLM_MODEL`. This usually means appending `us.*` (or other region) to the model ID (e.g., `us.anthropic.claude-3-5-sonnet-20241022-v2:0`). See [AWS Bedrock models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for available models.

Activate the Python environment:
```
source .venv/bin/activate
```

## Run the Default Pipeline

This script runs the cognify pipeline with default settings. It ingests text data, builds a knowledge graph, and allows you to run search queries.

```
python src/pipelines/default.py
```

## Run the Low-Level Pipeline

This script implements its own pipeline with custom ingestion task. It processes the given JSON data about companies and employees, making it searchable via a graph.

```
python src/pipelines/low_level.py
```

## Run the Custom Model Pipeline

Custom model uses custom pydantic model for graph extraction. This script categorizes programming languages as an example and visualizes relationships.

```
python src/pipelines/custom-model.py
```

## Graph preview 

cognee provides a visualize_graph function that will render the graph for you.

```
    graph_file_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".artifacts/graph_visualization.html")
        ).resolve()
    )
    await visualize_graph(graph_file_path)
```

# What will you build with cognee?

- Expand the dataset by adding more structured/unstructured data
- Customize the data model to fit your use case
- Use the search API to build an intelligent assistant
- Visualize knowledge graphs for better insights

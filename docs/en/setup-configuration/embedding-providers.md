# Embedding Providers

> Configure embedding providers for semantic search in Cognee

Embedding providers convert text into vector representations that enable semantic search. These vectors capture the meaning of text, allowing Cognee to find conceptually related content even when the wording is different.

<Info>
  **New to configuration?**

  See the [Setup Configuration Overview](./overview) for the complete workflow:

  install extras → create `.env` → choose providers → handle pruning.
</Info>

## Supported Providers

Cognee supports multiple embedding providers:

* **OpenAI** — Text embedding models via OpenAI API (default)
* **Azure OpenAI** — Text embedding models via Azure OpenAI Service
* **Google Gemini** — Embedding models via Google AI
* **Mistral** — Embedding models via Mistral AI
* **AWS Bedrock** — Embedding models via AWS Bedrock
* **Ollama** — Local embedding models via Ollama
* **LM Studio** — Local embedding models via LM Studio
* **Fastembed** — CPU-friendly local embeddings
* **Custom** — OpenAI-compatible embedding endpoints

<Warning>
  **LLM/Embedding Configuration**: If you configure only LLM or only embeddings, the other defaults to OpenAI. Ensure you have a working OpenAI API key, or configure both LLM and embeddings to avoid unexpected defaults.
</Warning>

## Configuration

<Accordion title="Environment Variables">
  Set these environment variables in your `.env` file:

  * `EMBEDDING_PROVIDER` — The provider to use (openai, gemini, mistral, ollama, fastembed, custom)
  * `EMBEDDING_MODEL` — The specific embedding model to use
  * `EMBEDDING_DIMENSIONS` — The vector dimension size (must match your vector store)
  * `EMBEDDING_API_KEY` — Your API key (falls back to `LLM_API_KEY` if not set)
  * `EMBEDDING_ENDPOINT` — Custom endpoint URL (for Azure, Ollama, or custom providers)
  * `EMBEDDING_API_VERSION` — API version (for Azure OpenAI)
  * `EMBEDDING_MAX_TOKENS` — Maximum tokens per request (optional)
</Accordion>

## Provider Setup Guides

<AccordionGroup>
  <Accordion title="OpenAI (Default)">
    OpenAI provides high-quality embeddings with good performance.

    ```dotenv  theme={null}
    EMBEDDING_PROVIDER="openai"
    EMBEDDING_MODEL="openai/text-embedding-3-large"
    EMBEDDING_DIMENSIONS="3072"
    # Optional
    # EMBEDDING_API_KEY=sk-...   # falls back to LLM_API_KEY if omitted
    # EMBEDDING_ENDPOINT=https://api.openai.com/v1
    # EMBEDDING_API_VERSION=
    # EMBEDDING_MAX_TOKENS=8191
    ```
  </Accordion>

  <Accordion title="Azure OpenAI Embeddings">
    Use Azure OpenAI Service for embeddings with your own deployment.

    ```dotenv  theme={null}
    EMBEDDING_PROVIDER="openai"
    EMBEDDING_MODEL="azure/text-embedding-3-large"
    EMBEDDING_ENDPOINT="https://<your-az>.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large"
    EMBEDDING_API_KEY="az-..."
    EMBEDDING_API_VERSION="2023-05-15"
    EMBEDDING_DIMENSIONS="3072"
    ```
  </Accordion>

  <Accordion title="Google Gemini">
    Use Google's embedding models for semantic search.

    ```dotenv  theme={null}
    EMBEDDING_PROVIDER="gemini"
    EMBEDDING_MODEL="gemini/text-embedding-004"
    EMBEDDING_API_KEY="AIza..."
    EMBEDDING_DIMENSIONS="768"
    ```
  </Accordion>

  <Accordion title="Mistral">
    Use Mistral's embedding models for high-quality vector representations.

    ```dotenv  theme={null}
    EMBEDDING_PROVIDER="mistral"
    EMBEDDING_MODEL="mistral/mistral-embed"
    EMBEDDING_API_KEY="sk-mis-..."
    EMBEDDING_DIMENSIONS="1024"
    ```

    **Installation**: Install the required dependency:

    ```bash  theme={null}
    pip install mistral-common[sentencepiece]
    ```
  </Accordion>

  <Accordion title="AWS Bedrock">
    Use embedding models provided by the AWS Bedrock service.

    ```dotenv  theme={null}
    EMBEDDING_PROVIDER="bedrock"
    EMBEDDING_MODEL="<your_model_name>"
    EMBEDDING_DIMENSIONS="<dimensions_of_the_model>"
    EMBEDDING_API_KEY="<your_api_key>"
    EMBEDDING_MAX_TOKENS="<max_tokens_of_your_model>"
    ```
  </Accordion>

  <Accordion title="Ollama (Local)">
    Run embedding models locally with Ollama for privacy and cost control.

    ```dotenv  theme={null}
    EMBEDDING_PROVIDER="ollama"
    EMBEDDING_MODEL="nomic-embed-text:latest"
    EMBEDDING_ENDPOINT="http://localhost:11434/api/embed"
    EMBEDDING_DIMENSIONS="768"
    HUGGINGFACE_TOKENIZER="nomic-ai/nomic-embed-text-v1.5"
    ```

    **Installation**: Install Ollama from [ollama.ai](https://ollama.ai) and pull your desired embedding model:

    ```bash  theme={null}
    ollama pull nomic-embed-text:latest
    ```
  </Accordion>

  <Accordion title="LM Studio (Local)">
    Run embedding models locally with LM Studio for privacy and cost control.

    ```dotenv  theme={null}
    EMBEDDING_PROVIDER="custom"
    EMBEDDING_MODEL="lm_studio/text-embedding-nomic-embed-text-1.5"
    EMBEDDING_ENDPOINT="http://127.0.0.1:1234/v1"
    EMBEDDING_API_KEY="."
    EMBEDDING_DIMENSIONS="768"
    ```

    **Installation**: Install LM Studio from [lmstudio.ai](https://lmstudio.ai/) and download your desired model from
    LM Studio's interface.
    Load your model, start the LM Studio server, and Cognee will be able to connect to it.
  </Accordion>

  <Accordion title="Fastembed (Local)">
    Use Fastembed for CPU-friendly local embeddings without GPU requirements.

    ```dotenv  theme={null}
    EMBEDDING_PROVIDER="fastembed"
    EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS="384"
    ```

    **Installation**: Fastembed is included by default with Cognee.

    **Known Issues**:

    * As of September 2025, Fastembed requires Python \< 3.13 (not compatible with Python 3.13+)
  </Accordion>

  <Accordion title="Custom Providers">
    Use OpenAI-compatible embedding endpoints from other providers.

    ```dotenv  theme={null}
    EMBEDDING_PROVIDER="custom"
    EMBEDDING_MODEL="provider/your-embedding-model"
    EMBEDDING_ENDPOINT="https://your-endpoint.example.com/v1"
    EMBEDDING_API_KEY="provider-..."
    EMBEDDING_DIMENSIONS="<match-your-model>"
    ```
  </Accordion>
</AccordionGroup>

## Advanced Options

<Accordion title="Rate Limiting">
  ```dotenv  theme={null}
  EMBEDDING_RATE_LIMIT_ENABLED="true"
  EMBEDDING_RATE_LIMIT_REQUESTS="10"
  EMBEDDING_RATE_LIMIT_INTERVAL="5"
  ```
</Accordion>

<Accordion title="Testing and Development">
  ```dotenv  theme={null}
  # Mock embeddings for testing (returns zero vectors)
  MOCK_EMBEDDING="true"
  ```
</Accordion>

## Important Notes

* **Dimension Consistency**: `EMBEDDING_DIMENSIONS` must match your vector store collection schema
* **API Key Fallback**: If `EMBEDDING_API_KEY` is not set, Cognee uses `LLM_API_KEY` (except for custom providers)
* **Tokenization**: For Ollama and Hugging Face models, set `HUGGINGFACE_TOKENIZER` for proper token counting
* **Performance**: Local providers (Ollama, Fastembed) are slower but offer privacy and cost benefits

<Columns cols={3}>
  <Card title="LLM Providers" icon="brain" href="/setup-configuration/llm-providers">
    Configure LLM providers for text generation
  </Card>

  <Card title="Vector Stores" icon="database" href="/setup-configuration/vector-stores">
    Set up vector databases for embedding storage
  </Card>

  <Card title="Overview" icon="settings" href="/setup-configuration/overview">
    Return to setup configuration overview
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
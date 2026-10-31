# Supported Ollama Models for Structured Graph Extraction

Cognee supports using local Large Language Models (LLMs) via Ollama. However, because Cognee relies on structured output generation (using Instructor with JSON schemas) to extract knowledge graphs, the performance and reliability of the extraction pipeline depend heavily on the model's capabilities.

This guide lists recommended models, models with known limitations, and troubleshooting tips.

---

## Self-hosted setup (Ollama / OpenAI-compatible endpoints)

Use these environment variables for local or self-hosted models. Copy them into `.env` (see `.env.template`) or set them before importing `cognee`.

### Ollama (recommended local path)

1. Start Ollama and pull a validated model plus embeddings:
   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
2. Configure Cognee:
   ```bash
   LLM_PROVIDER=ollama
   LLM_MODEL=llama3.1:8b
   LLM_ENDPOINT=http://localhost:11434
   LLM_API_KEY=ollama
   LLM_TEMPERATURE=0.0

   EMBEDDING_PROVIDER=ollama
   EMBEDDING_MODEL=nomic-embed-text
   EMBEDDING_ENDPOINT=http://localhost:11434/api/embed
   EMBEDDING_DIMENSIONS=768
   HUGGINGFACE_TOKENIZER=nomic-ai/nomic-embed-text-v1.5
   ```
3. Set `LLM_ENDPOINT` based on `STRUCTURED_OUTPUT_FRAMEWORK` (default is `litellm_native`):
   - **`litellm_native` (default):** `LLM_ENDPOINT=http://localhost:11434` — LiteLLM appends Ollama's native paths; a `/v1` suffix returns 404.
   - **`instructor`:** `LLM_ENDPOINT=http://localhost:11434/v1` and `STRUCTURED_OUTPUT_FRAMEWORK=instructor`.

A full runnable script is in [`examples/guides/local_ollama_example.py`](../examples/guides/local_ollama_example.py) (that example uses the OpenAI-compatible `/v1` endpoint — pair it with `STRUCTURED_OUTPUT_FRAMEWORK=instructor`, or change the endpoint to the bare host for the default `litellm_native` framework).

### vLLM or other OpenAI-compatible servers

Point Cognee at any OpenAI-compatible base URL (vLLM, llama.cpp server, LocalAI, etc.):

```bash
LLM_PROVIDER=openai
LLM_MODEL=openai/<your-served-model-id>
LLM_ENDPOINT=http://localhost:8000/v1
LLM_API_KEY=not-needed   # or whatever the server expects
LLM_TEMPERATURE=0.0

# Embeddings: keep Ollama / fastembed / a remote embedding API as you prefer
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_ENDPOINT=http://localhost:11434/api/embed
EMBEDDING_DIMENSIONS=768
```

Notes:

- `LLM_MODEL` must include the provider prefix LiteLLM expects (`openai/...` when `LLM_PROVIDER=openai`).
- Prefer a model from the recommended matrix below for reliable structured graph extraction.
- Optional per-stage overrides (`LLM_EXTRACTION_*`, `LLM_SUMMARIZATION_*`, `LLM_QUERY_*`) are documented in `.env.template` if extraction and query should use different endpoints.


## Model Support Matrix

### 1. Recommended / Validated Models
These models consistently format output correctly according to abstract JSON schemas, making them highly reliable for Cognee's graph extraction:

- **Llama 3.1 (8B, 70B)** (e.g., `llama3.1:8b`, `llama3.1:70b`) — **Highly Recommended**
- **Llama 3.2 (3B)** (e.g., `llama3.2:3b`) — Recommended for lightweight or resource-constrained environments.
- **Llama 3.3 (70B)** (e.g., `llama3.3`) — Outstanding extraction capability if hardware permits.
- **Qwen 2.5 (14B, 32B, 72B)** (e.g., `qwen2.5:14b`, `qwen2.5:32b`, `qwen2.5:72b`) — Strong extraction and reasoning capability.

### 2. Known Issues & Limitations
These models have high failure rates during structured JSON schema extraction. They often output invalid JSON, verbose conversational padding, or fail to follow abstract object definitions, leading to empty or dropped graphs:

- **Mistral (7B)** (e.g., `mistral`, `mistral:7b`) — Unstable structured JSON output, prone to schema format violations.
- **Phi 3 / Phi 3.5** (e.g., `phi3`, `phi3.5`) — Fails to consistently adhere to Pydantic schemas.
- **Qwen 2.5 (7B and smaller)** (e.g., `qwen2.5:7b`, `qwen2.5:3b`, `qwen2.5:1.5b`) — Struggles with complex schemas compared to the larger $14\text{B}+$ variants.
- **Gemma 2 (2B, 9B)** (e.g., `gemma2:2b`, `gemma2:9b`) — Prone to schema validation drops.

### 3. Unknown / Experimental Models
Any model not listed above is treated as unvalidated/experimental. If you choose to run an unvalidated model, Cognee will emit a warning but will **not** block execution.

---

## Troubleshooting Local Extraction

If you notice that `cognify()` is running but your final queries yield empty search results or no nodes are created, check the following:

1. **Verify your Model**: Ensure you are using one of the recommended models (e.g., `llama3.1:8b`).
2. **Check Temperature**: Extraction runs at temperature `0.0` by default on Ollama, for deterministic output formatting. If you set `LLM_TEMPERATURE` to something higher, lower it back to `0.0` in your `.env`.
3. **Verify API Connection**: Ensure Ollama (or your OpenAI-compatible server) is running and that `LLM_ENDPOINT` matches your `STRUCTURED_OUTPUT_FRAMEWORK` — see [Self-hosted setup](#self-hosted-setup-ollama--openai-compatible-endpoints) above.
4. **Inspect Logging**: Check the console log outputs. If Cognee catches validation errors during extraction, they will be reported as warnings.

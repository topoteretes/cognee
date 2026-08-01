# Supported Ollama Models for Structured Graph Extraction

Cognee supports using local Large Language Models (LLMs) via Ollama. However, because Cognee relies on structured output generation (using Instructor with JSON schemas) to extract knowledge graphs, the performance and reliability of the extraction pipeline depend heavily on the model's capabilities.

This guide lists recommended models, models with known limitations, and troubleshooting tips.

---

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
2. **Set Temperature to 0**: Keep `LLM_TEMPERATURE=0.0` (which is Cognee's default) to force deterministic output formatting.
3. **Verify API Connection**: Ensure Ollama is running and accessible (usually at `http://localhost:11434/v1`).
4. **Inspect Logging**: Check the console log outputs. If Cognee catches validation errors during extraction, they will be reported as warnings.

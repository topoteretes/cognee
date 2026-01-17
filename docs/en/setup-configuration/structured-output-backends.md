# Structured Output Backends

> Configure structured output frameworks for reliable data extraction in Cognee

Structured output backends ensure reliable data extraction from LLM responses. Cognee supports two frameworks that convert LLM text into structured Pydantic models for knowledge graph extraction and other tasks.

<Info>
  **New to configuration?**

  See the [Setup Configuration Overview](./overview) for the complete workflow:

  install extras → create `.env` → choose providers → handle pruning.
</Info>

## Supported Frameworks

Cognee supports two structured output approaches:

* **LiteLLM + Instructor** — Provider-agnostic client with Pydantic coercion (default)
* **BAML** — DSL-based framework with type registry and guardrails

Both frameworks produce the same Pydantic-validated outputs, so your application code remains unchanged regardless of which backend you choose.

## How It Works

Cognee uses a unified interface that abstracts the underlying framework:

```python  theme={null}
from cognee.infrastructure.llm.LLMGateway import LLMGateway
await LLMGateway.acreate_structured_output(text, system_prompt, response_model)
```

The `STRUCTURED_OUTPUT_FRAMEWORK` environment variable determines which backend processes your requests, but the API remains identical.

## Configuration

<Tabs>
  <Tab title="LiteLLM + Instructor (Default)">
    ```dotenv  theme={null}
    STRUCTURED_OUTPUT_FRAMEWORK=instructor
    ```
  </Tab>

  <Tab title="BAML">
    ```dotenv  theme={null}
    STRUCTURED_OUTPUT_FRAMEWORK=baml
    ```
  </Tab>
</Tabs>

## Important Notes

* **Unified Interface**: Your application code uses the same `acreate_structured_output()` call regardless of framework
* **Provider Flexibility**: Both frameworks support the same LLM providers
* **Output Consistency**: Both produce identical Pydantic-validated results
* **Performance**: Framework choice doesn't significantly impact performance

<Columns cols={3}>
  <Card title="LLM Providers" icon="brain" href="/setup-configuration/llm-providers">
    Configure LLM providers for text generation
  </Card>

  <Card title="Overview" icon="settings" href="/setup-configuration/overview">
    Return to setup configuration overview
  </Card>

  <Card title="Custom Prompts" icon="text-wrap" href="/guides/custom-prompts">
    Learn about custom prompt configuration
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
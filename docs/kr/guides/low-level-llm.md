# Low-Level LLM

> Step-by-step guide to using acreate_structured_output for direct LLM interaction

A minimal guide to the one function you can call directly to get Pydantic-validated structured output from an LLM.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have some text to process

## What It Is

* Single entrypoint: `LLMGateway.acreate_structured_output(text, system_prompt, response_model)`
* Returns an instance of your Pydantic `response_model` filled by the LLM
* Backend-agnostic: uses BAML or LiteLLM+Instructor under the hood based on config — your code doesn't change

<Note>
  This function is used by default during cognify via the extractor. The backend switch lives in `cognee/infrastructure/llm/LLMGateway.py`.
</Note>

## Code in Action

```python  theme={null}
import asyncio

from pydantic import BaseModel
from typing import List
from cognee.infrastructure.llm.LLMGateway import LLMGateway

class MiniEntity(BaseModel):
    name: str
    type: str

class MiniGraph(BaseModel):
    nodes: List[MiniEntity]

async def main():

    system_prompt = (
        "Extract entities as nodes with name and type. "
        "Use concise, literal values present in the text."
    )

    text = "Apple develops iPhone; Audi produces the R8."

    result = await LLMGateway.acreate_structured_output(text, system_prompt, MiniGraph)
    print(result)
    # MiniGraph(nodes=[MiniEntity(name='Apple', type='Organization'), ...])

if __name__ == "__main__":
    asyncio.run(main())
```

<Note>
  This simple example uses a basic schema for demonstration. In practice, you can define complex Pydantic models with nested structures, validation rules, and custom types.
</Note>

## What Just Happened

### Step 1: Define Your Schema

```python  theme={null}
class MiniEntity(BaseModel):
    name: str
    type: str

class MiniGraph(BaseModel):
    nodes: List[MiniEntity]
```

Create Pydantic models that define the structure you want the LLM to return. The LLM will fill these models with data extracted from your text.

### Step 2: Write a System Prompt

```python  theme={null}
system_prompt = (
    "Extract entities as nodes with name and type. "
    "Use concise, literal values present in the text."
)
```

Write a clear prompt that tells the LLM what to extract and how to structure it. Short, explicit prompts work best.

### Step 3: Call the LLM

```python  theme={null}
result = await LLMGateway.acreate_structured_output(text, system_prompt, MiniGraph)
```

This calls the LLM with your text and prompt, returning a Pydantic model instance with the extracted data.

<Tip>
  A sync variant exists: `LLMGateway.create_structured_output(...)`.
</Tip>

## Custom Tasks

This function is often used when creating custom tasks for processing data with structured output. You'll see it in action when we cover custom task creation in a future guide.

## Backend Doesn't Matter

The config decides the engine:

* `STRUCTURED_OUTPUT_FRAMEWORK=instructor` → LiteLLM + Instructor
* `STRUCTURED_OUTPUT_FRAMEWORK=baml` → BAML client/registry

Both paths return the same Pydantic model instance to your code.

<Columns cols={3}>
  <Card title="Structured Output" icon="brackets" href="/setup-configuration/structured-output-backends">
    Learn about structured output frameworks
  </Card>

  <Card title="Custom Prompts" icon="text-wrap" href="/guides/custom-prompts">
    Control extraction with custom prompts
  </Card>

  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Explore API endpoints
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
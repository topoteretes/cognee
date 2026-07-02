# LLM-mocked tests for Cognee examples

Tests here run every `examples/` script with mocked LLM + embedding calls — no API keys, no network, fully deterministic.

## How it works

`conftest.py` provides a **session-scoped pytest fixture** `mock_llm_and_embeddings` that patches two layers:

| Layer | What's patched | Returns |
|---|---|---|
| LLM | `LLMGateway.acreate_structured_output` | Minimal valid Pydantic instance of whatever `response_model` is requested |
| Embeddings | `LiteLLMEmbeddingEngine.embed_text` | Fixed 64-dim vector `[0.1, 0.1, …]` for every text chunk |

## Folder structure

```
cognee/tests/examples/
├── conftest.py              ← shared mock harness (edit this to add new model stubs)
├── README.md                ← this file
├── guides/
│   └── test_guides_mocked.py
├── demos/                   ← add tests here when covering examples/demos/
└── ...
```

## Adding a test for a new example

1. Identify which folder the example lives in under `examples/`.
2. Create or open the matching test file under `cognee/tests/examples/<folder>/`.
3. Write a test that:
   - Accepts the `mock_llm_and_embeddings` fixture (injected automatically by pytest).
   - Imports and runs the example's `main()` coroutine.
   - Asserts no exception is raised.

```python
@pytest.mark.asyncio
async def test_my_new_example(mock_llm_and_embeddings):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "my_example", Path(__file__).parents[4] / "examples" / "guides" / "my_example.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    await mod.main()
```

4. If the example calls `visualize_graph()`, patch it to avoid filesystem writes:
   ```python
   with patch("cognee.visualize_graph", new=AsyncMock(return_value="/tmp/mock.html")):
       await mod.main()
   ```

5. If the example asserts on recall results (`assert result != []`), patch `cognee.recall` to return a non-empty list:
   ```python
   with patch("cognee.recall", new=AsyncMock(return_value=["mock result"])):
       await mod.main()
   ```

## Running the suite

```bash
# From the repo root
pytest cognee/tests/examples/ -v
```

No environment variables or API keys needed.

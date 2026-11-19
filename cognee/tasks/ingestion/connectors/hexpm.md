# Hex.pm (Elixir / Erlang) Connector

The Hex.pm connector allows Cognee to ingest Elixir and Erlang package metadata, hexdocs documentation references, and dependency definitions into knowledge memory.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors.hexpm import hexpm_source

# Ingest Elixir dependencies into technical memory
await cognee.remember(
    hexpm_source(packages=["phoenix", "ecto", "jason", "plug"]),
    dataset_name="elixir_dependencies"
)
```

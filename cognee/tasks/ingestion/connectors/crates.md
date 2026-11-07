# Crates.io (Rust) Connector

The Crates.io connector allows Cognee to ingest Rust crate metadata, documentation links, and crate dependencies into knowledge memory.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors.crates import crates_source

# Ingest Rust dependencies into technical memory
await cognee.remember(
    crates_source(crates=["tokio", "serde", "anyhow", "axum"]),
    dataset_name="rust_dependencies"
)
```

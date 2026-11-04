# Packagist / Composer Connector

The Packagist connector enables Cognee to ingest PHP and Composer package specifications, version hierarchies, dependencies, and maintainer metadata into knowledge memory.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors.packagist import packagist_source

# Ingest PHP dependencies into technical memory
await cognee.remember(
    packagist_source(packages=["laravel/framework", "guzzlehttp/guzzle", "monolog/monolog"]),
    dataset_name="php_dependencies"
)
```

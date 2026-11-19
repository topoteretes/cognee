# NuGet v3 (.NET) Connector

The NuGet connector allows Cognee to ingest .NET and C# package specifications, target framework details, and project metadata into knowledge memory.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors.nuget import nuget_source

# Ingest .NET dependencies into technical memory
await cognee.remember(
    nuget_source(packages=["Newtonsoft.Json", "Serilog", "Microsoft.EntityFrameworkCore"]),
    dataset_name="dotnet_dependencies"
)
```

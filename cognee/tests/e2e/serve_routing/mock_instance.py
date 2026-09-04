"""A real cognee API instance with a mocked LLM, launched as a subprocess.

Run as ``python -m cognee.tests.e2e.serve_routing.mock_instance <root> <port>``.

The server is real — real FastAPI app, real routes, real Ladybug/LanceDB/SQLite
— so the HTTP contracts under test are the production ones. Only the LLM is
replaced (embeddings go through the built-in ``MOCK_EMBEDDING`` flag), which
keeps the suite offline, deterministic, and runnable without API keys.

The knowledge graph follows the document text, so a document updated from one
version to another produces visibly different entities. That is what lets the
suite assert that an update really replaced the stored content.
"""

import os
import sys

ROOT = sys.argv[1]
PORT = int(sys.argv[2])

# Hermetic by construction: cognee's import runs dotenv.load_dotenv(override=True),
# which on a developer machine would pull the repo .env over everything set here
# (a real Postgres, access control on) and silently point this test at live data.
# CI has no .env, but the suite must behave identically in both places.
import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: False
dotenv.main.load_dotenv = lambda *args, **kwargs: False

os.environ.update(
    DATA_ROOT_DIRECTORY=os.path.join(ROOT, "data"),
    SYSTEM_ROOT_DIRECTORY=os.path.join(ROOT, "system"),
    DB_PROVIDER="sqlite",
    DB_NAME="serve_routing.db",
    VECTOR_DB_PROVIDER="lancedb",
    GRAPH_DATABASE_PROVIDER="ladybug",
    MOCK_EMBEDDING="true",
    TELEMETRY_DISABLED="1",
    COGNEE_SKIP_CONNECTION_TEST="true",
    LLM_API_KEY="sk-mocked-never-called",
    # Auth ON, like a real deployment: the suite drives the instance with an
    # API key over X-Api-Key, which is the header the cloud path uses. Testing
    # the proxy against an unauthenticated instance would skip the layer most
    # likely to break a proxied call.
    ENABLE_BACKEND_ACCESS_CONTROL="true",
    # Pin the default account rather than inheriting DEFAULT_USER_EMAIL /
    # DEFAULT_USER_PASSWORD from the environment: the suite logs in with these
    # to mint its API key, so the instance has to be self-describing.
    DEFAULT_USER_EMAIL="serve-routing-e2e@example.com",
    DEFAULT_USER_PASSWORD="serve-routing-e2e-password",
)

from cognee.infrastructure.llm.LLMGateway import LLMGateway  # noqa: E402
from cognee.shared.data_models import (  # noqa: E402
    Edge,
    KnowledgeGraph,
    Node,
    SummarizedContent,
)

# Entities are keyed off marker words in the text, so v1 and v2 of a document
# yield different graphs and "did the update actually replace the content?" is
# an observable question.
MARKERS = ("Berlin", "Bordeaux", "cartographer", "sommelier", "Alice")


@staticmethod
async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
    if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
        found = [m for m in MARKERS if m in str(text_input)]
        nodes = [
            Node(id=m, name=m, type="Marker", description=f"marker {m}", label=m) for m in found
        ]
        edges = [
            Edge(source_node_id=found[0], target_node_id=other, relationship_name="mentions")
            for other in found[1:]
        ]
        return KnowledgeGraph(summary="s", description="s", nodes=nodes, edges=edges)
    if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
        return SummarizedContent(summary="Mock summary.", description="")
    if response_model is str:
        return "mock answer"
    return response_model()


LLMGateway.acreate_structured_output = _mock_acreate

import uvicorn  # noqa: E402
from cognee.api.client import app  # noqa: E402

uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

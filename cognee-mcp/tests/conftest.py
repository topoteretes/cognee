import pathlib
import pytest
import cognee
from src import server
from src.cognee_client import CogneeClient


@pytest.fixture(autouse=True)
async def setup_isolated_cognee(request):
    test_name = request.node.name
    test_base = pathlib.Path(request.fspath).parent

    cognee.config.data_root_directory(str(test_base / f".data_storage/{test_name}"))
    cognee.config.system_root_directory(str(test_base / f".cognee_system/{test_name}"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    server.cognee_client = CogneeClient(api_url=None, api_token=None)

    yield

    server.cognee_client = None

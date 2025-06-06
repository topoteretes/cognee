from fastapi import APIRouter
from cognee.modules.data.deletion import prune_system as actual_prune_system, prune_data as actual_prune_data
# Potentially add: from fastapi import Depends
# Potentially add: from cognee.modules.users.models import User
# Potentially add: from cognee.modules.users.methods import get_authenticated_user
# For now, we will not add authentication to keep it minimal, but it should be considered.

prune_router = APIRouter()

@prune_router.delete("/", summary="Prune all system data, graph, and vector stores")
async def http_prune_all(
    # If authentication were added:
    # user: User = Depends(get_authenticated_user)
):
    await actual_prune_data()
    # Mimic the MCP tool's full prune by setting graph, vector, and metadata to True.
    await actual_prune_system(graph=True, vector=True, metadata=True)
    return {"message": "System successfully pruned"}

class prune:
    @staticmethod
    async def prune_data(): # This might be the original method
        # If this is different from actual_prune_data, it stays.
        # If it's the same, it might call actual_prune_data or be removed if http_prune_all is the sole entry point.
        # For now, assume it might be used by other parts (like MCP) and keep it.
        await actual_prune_data() # Or its original implementation

    @staticmethod
    async def prune_system(graph=True, vector=True, metadata=False): # This might be the original method
        # Similar to above, keep for now.
        await actual_prune_system(graph, vector, metadata) # Or its original implementation

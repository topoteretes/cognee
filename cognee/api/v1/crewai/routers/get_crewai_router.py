from fastapi import APIRouter, Depends
from cognee.api.DTO import InDTO
from cognee.modules.users.get_fastapi_users import get_fastapi_users
from cognee.modules.users.authentication.get_auth_backend import get_auth_backend
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


class CrewAIRunPayloadDTO(InDTO):
    username1: str
    username2: str

def get_crewai_router() -> APIRouter:
    router = APIRouter()

    @router.post("/run", response_model=str)
    async def run_crewai(payload: CrewAIRunPayloadDTO, user: User = Depends(get_authenticated_user)):
        # Run CrewAI with the provided usernames
        print(payload.username1, payload.username2)

        return "CrewAI run started"

    return router

from fastapi import APIRouter, Depends, HTTPException
from pydantic import EmailStr

from cognee.api.DTO import InDTO
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.methods.get_user_id_by_email import get_user_id_by_email
from cognee.modules.users.models.User import User


class UserEmailRequest(InDTO):
    email: EmailStr


def get_user_id_by_email_router() -> APIRouter:
    router = APIRouter()

    @router.post("/get-user-id")
    async def get_user_id(body: UserEmailRequest, user: User = Depends(get_authenticated_user)):
        """Look up a user id by email address.

        Returns user_id as a string, for sharing. Only users the caller may see resolve: the
        caller themselves, the members of the caller's tenants, or, for a caller in no tenant,
        users in no tenant. Any other address returns 404 whether or not it exists; a value
        that is not a valid email address returns 400.

        ## Request Parameters
        - **email** (EmailStr): Email address of the user.
        """
        user_id = await get_user_id_by_email(str(body.email), user)

        if user_id is None:
            raise HTTPException(status_code=404, detail="User not found")

        return {"user_id": str(user_id)}

    return router

from uuid import UUID
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import Field
from typing import List, Optional
from fastapi import APIRouter, Depends

from cognee.api.DTO import InDTO
from cognee.infrastructure.databases.relational import get_async_session
from cognee.infrastructure.utils.run_async import run_async
from cognee.modules.notebooks.models import Notebook, NotebookCell
from cognee.modules.notebooks.operations import run_in_local_sandbox
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.notebooks.methods import (
    create_notebook,
    delete_notebook,
    get_notebook,
    get_notebooks,
    update_notebook,
)


class NotebookData(InDTO):
    name: Optional[str] = Field(...)
    cells: Optional[List[NotebookCell]] = Field(default=[])


def get_notebooks_router():
    router = APIRouter()

    @router.get("")
    async def get_notebooks_endpoint(user: User = Depends(get_authenticated_user)):
        async with get_async_session() as session:
            return await get_notebooks(user.id, session)

    @router.post("")
    async def create_notebook_endpoint(
        notebook_data: NotebookData, user: User = Depends(get_authenticated_user)
    ):
        return await create_notebook(
            user.id, notebook_data.name, notebook_data.cells, deletable=True
        )

    @router.put("/{notebook_id}")
    async def update_notebook_endpoint(
        notebook_id: UUID, notebook_data: NotebookData, user: User = Depends(get_authenticated_user)
    ):
        async with get_async_session(auto_commit=True) as session:
            notebook: Notebook = await get_notebook(notebook_id, user.id, session)

            if notebook is None:
                return JSONResponse(status_code=404, content={"error": "Notebook not found"})

            if notebook_data.name and notebook_data.name != notebook.name:
                notebook.name = notebook_data.name

            if notebook_data.cells:
                notebook.cells = notebook_data.cells

            return await update_notebook(notebook, session)

    class RunCodeData(InDTO):
        content: str = Field(...)

    @router.post("/{notebook_id}/{cell_id}/run")
    async def run_notebook_cell_endpoint(
        notebook_id: UUID,
        cell_id: UUID,
        run_code: RunCodeData,
        user: User = Depends(get_authenticated_user),
    ):
        async with get_async_session() as session:
            notebook: Notebook = await get_notebook(notebook_id, user.id, session)

            if notebook is None:
                return JSONResponse(status_code=404, content={"error": "Notebook not found"})

            result, error = await run_async(run_in_local_sandbox, run_code.content)

            return JSONResponse(
                status_code=200, content={"result": jsonable_encoder(result), "error": error}
            )

    @router.delete("/{notebook_id}")
    async def delete_notebook_endpoint(
        notebook_id: UUID, user: User = Depends(get_authenticated_user)
    ):
        async with get_async_session(auto_commit=True) as session:
            notebook: Notebook = await get_notebook(notebook_id, user.id, session)

            if notebook is None:
                return JSONResponse(status_code=404, content={"error": "Notebook not found"})

            return await delete_notebook(notebook, session)

    return router

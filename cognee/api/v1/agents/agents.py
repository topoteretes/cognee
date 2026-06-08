from __future__ import annotations

from typing import Optional, Union
from uuid import UUID

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods import give_permission_on_dataset
from cognee.modules.data.methods import get_authorized_dataset, get_datasets_by_name
from cognee.modules.agents.create_agent import create_agent
from cognee.modules.agents.list_agents import list_agents
from cognee.modules.agents.get_agent import get_agent
from cognee.modules.agents.delete_agent import delete_agent
from cognee.modules.agents.agent_mode import register_agent, unregister_agent
from cognee.modules.agents.operations import (
    RangeLiteral,
    list_agent_connections,
    get_agent_connection_detail,
)
from cognee.modules.agents.models import (
    AgentConnectionType,
    AgentMemoryMode,
    AgentSource,
    RegisterAgentRequest,
    UnregisterAgentRequest,
)


def _display_email(email: str) -> str:
    """Convert an internal agent email to its display form.

    Internal agent emails are minted as ``{slug}+{parent_id}@cognee.agent``;
    this strips the ``+{parent_id}`` segment to produce ``{slug}@cognee.agent``.
    """
    local, _, domain = email.partition("@")
    slug = local.split("+", 1)[0]
    return f"{slug}@{domain}"


class agents:
    """
    Agent management namespace for Cognee.

    All methods are static and provide operations for creating, listing,
    inspecting, and deleting agents, as well as registering and listing
    agent connections. Permission scoping is enforced per acting user.

    Example:
        ```python
        import cognee

        # Create an agent with read/write on a dataset
        agent = await cognee.agents.create("my-agent", datasets=["my_project"])

        # List your agents
        all_agents = await cognee.agents.list()

        # Register an agent connection
        connection = await cognee.agents.register("session-1")
        ```
    """

    @staticmethod
    async def create(
        name: str,
        datasets: Optional[list[Union[str, UUID]]] = None,
        user: Optional[User] = None,
    ) -> dict:
        if user is None:
            user = await get_default_user()

        # Resolve and authorize EVERY requested dataset before minting the agent.
        # Doing this first means a failed authorization never leaves an orphaned
        # agent user with a live API key (and no partial dataset grants).
        authorized_dataset_ids: list[UUID] = []
        for entry in datasets or []:
            # Resolve the entry to a dataset id (UUID).
            try:
                dataset_id = UUID(str(entry))
            except ValueError:
                matched = await get_datasets_by_name(str(entry), user.id)
                if not matched:
                    raise ValueError(f"Dataset '{entry}' not found.")
                dataset_id = UUID(str(matched[0].id))

            # Verify the CALLING user is authorized read on the dataset BEFORE
            # granting the new agent any access.
            authorized_dataset = await get_authorized_dataset(user, dataset_id, "read")
            if authorized_dataset is None:
                raise PermissionDeniedError(f"Dataset {dataset_id} not accessible.")
            authorized_dataset_ids.append(dataset_id)

        agent_user, agent_api_key = await create_agent(name, user)

        # Grant the new agent user read and write on each authorized dataset.
        for dataset_id in authorized_dataset_ids:
            await give_permission_on_dataset(agent_user, dataset_id, "read")
            await give_permission_on_dataset(agent_user, dataset_id, "write")

        return {
            "agent_id": str(agent_user.id),
            "agent_email": _display_email(agent_user.email),
            "agent_api_key": agent_api_key,
        }

    @staticmethod
    async def list(user: Optional[User] = None) -> list[dict]:
        if user is None:
            user = await get_default_user()

        infos = await list_agents(user.id)

        return [
            {
                "agent_id": str(info.user.id),
                "agent_email": _display_email(info.user.email),
                "api_key_label": info.api_key_label,
            }
            for info in infos
        ]

    @staticmethod
    async def get(agent_id: Union[str, UUID], user: Optional[User] = None) -> dict:
        if user is None:
            user = await get_default_user()

        agent_uuid = agent_id if isinstance(agent_id, UUID) else UUID(str(agent_id))

        try:
            info = await get_agent(agent_uuid, user.id)
        except LookupError:
            raise ValueError(f"Agent {agent_id} not found")
        except PermissionError:
            raise PermissionDeniedError("Not authorized to view this agent")

        return {
            "agent_id": str(info.user.id),
            "agent_email": _display_email(info.user.email),
            "api_key_label": info.api_key_label,
        }

    @staticmethod
    async def delete(agent_id: Union[str, UUID], user: Optional[User] = None) -> None:
        if user is None:
            user = await get_default_user()

        agent_uuid = agent_id if isinstance(agent_id, UUID) else UUID(str(agent_id))

        try:
            await delete_agent(agent_uuid, user.id)
        except LookupError:
            raise ValueError(f"Agent {agent_id} not found")
        except PermissionError:
            raise PermissionDeniedError("Not authorized to delete this agent")

    @staticmethod
    async def register(
        agent_session_name: str,
        user: Optional[User] = None,
        type: AgentConnectionType = "api",
        memory_mode: AgentMemoryMode = "unknown",
        session_id: Optional[str] = None,
        dataset_ids: Optional[list[str]] = None,
        dataset_names: Optional[list[str]] = None,
        source: AgentSource = "api",
        origin_function: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        if user is None:
            user = await get_default_user()

        # Validate the calling user is authorized read on every supplied dataset
        # BEFORE building/sending the register request.
        for did in dataset_ids or []:
            dataset = await get_authorized_dataset(user, UUID(str(did)), "read")
            if dataset is None:
                raise PermissionDeniedError(f"Dataset {did} not accessible.")

        for dname in dataset_names or []:
            matched = await get_datasets_by_name(dname, user.id)
            if not matched:
                raise PermissionDeniedError(f"Dataset '{dname}' not accessible.")
            dataset = await get_authorized_dataset(user, UUID(str(matched[0].id)), "read")
            if dataset is None:
                raise PermissionDeniedError(f"Dataset '{dname}' not accessible.")

        request = RegisterAgentRequest(
            agent_session_name=agent_session_name,
            type=type,
            memory_mode=memory_mode,
            session_id=session_id,
            dataset_ids=dataset_ids or [],
            dataset_names=dataset_names or [],
            source=source,
            origin_function=origin_function,
            metadata=metadata or {},
        )

        connection = await register_agent(user, request)

        return connection.model_dump(mode="json")

    @staticmethod
    async def unregister(agent_session_name: str, user: Optional[User] = None) -> int:
        if user is None:
            user = await get_default_user()

        request = UnregisterAgentRequest(agent_session_name=agent_session_name)

        return await unregister_agent(user, request)

    @staticmethod
    async def list_connections(
        user: Optional[User] = None,
        agent_id: Optional[Union[str, UUID]] = None,
        range_key: RangeLiteral = "30d",
        status_filter: Optional[str] = None,
        include_sources: bool = True,
        active_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        if user is None:
            user = await get_default_user()

        if agent_id is not None and not isinstance(agent_id, UUID):
            agent_id = UUID(str(agent_id))

        response = await list_agent_connections(
            user=user,
            agent_id=agent_id,
            range_key=range_key,
            status_filter=status_filter,
            include_sources=include_sources,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

        # Defensive scope filter: the backend visibility rule treats in-memory
        # registered connections with no owning user AND no datasets as visible to
        # everyone. Drop them here so a caller never sees connections that are not
        # bound to their own user scope or a dataset they can read.
        scoped_agents = [
            agent for agent in response.agents if agent.user_id is not None or agent.datasets
        ]
        removed = len(response.agents) - len(scoped_agents)
        if removed:
            response.agents = scoped_agents
            response.total = max(0, response.total - removed)
            response.has_more = response.offset + len(scoped_agents) < response.total

        return response.model_dump(mode="json")

    @staticmethod
    async def get_connection(
        agent_id: Union[str, UUID],
        user: Optional[User] = None,
        agent_session_name: Optional[str] = None,
    ) -> Optional[dict]:
        if user is None:
            user = await get_default_user()

        agent_uuid = agent_id if isinstance(agent_id, UUID) else UUID(str(agent_id))

        detail = await get_agent_connection_detail(
            user=user,
            agent_id=agent_uuid,
            agent_session_name=agent_session_name,
        )

        return detail.model_dump(mode="json") if detail is not None else None

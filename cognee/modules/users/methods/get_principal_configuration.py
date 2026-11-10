from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods.get_visible_user_ids import get_visible_user_ids
from cognee.modules.users.models.PrincipalConfiguration import PrincipalConfiguration


async def get_principal_configuration(config_id: UUID, principal_id: UUID) -> dict:
    """
    Retrieves one stored Cognee configuration by its id, as seen by ``principal_id``.

    The lookup is scoped to the principals ``principal_id`` can see: itself and
    the agent users it is the parent of (``get_visible_user_ids``), so a human
    user can read the configuration of an agent they provisioned. Visibility runs
    downward only — an agent cannot read its parent's configuration. Any other
    owner's configuration is treated exactly like a missing one, so a caller
    cannot read, or probe for, configurations that are not theirs.

    Args:
        config_id (UUID): The unique identifier of the config.
        principal_id (UUID): The caller; the config must belong to it or to one of its agents.

    Returns:
        dict: The configuration data if visible to this principal, otherwise an empty dictionary.
    """
    visible_owner_ids = await get_visible_user_ids(principal_id)
    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        query = select(PrincipalConfiguration).where(
            PrincipalConfiguration.id == config_id,
            PrincipalConfiguration.owner_id.in_(visible_owner_ids),
        )

        result = await session.execute(query)
        config_record = result.scalars().first()

        # Return the configuration dictionary if the record exists, otherwise an empty dict
        return config_record.configuration if config_record else {}


async def get_principal_all_configuration(principal_id: UUID) -> list[dict[str, dict]]:
    """
    Retrieves all Cognee configurations owned by exactly one principal.

    Deliberately not widened to child agents: the agent registry and the
    integrations surfaces call this per agent user and rely on getting that
    agent's records only. The configuration router widens over
    ``get_visible_user_ids`` itself.

    Args:
        principal_id (UUID): The unique identifier of the owner (user/group).

    Returns:
        list[dict]: A list of configuration dictionaries. Returns an empty list if none are found.
    """
    relational_engine = get_relational_engine()

    async with relational_engine.get_async_session() as session:
        # Select all records belonging to the principal
        query = select(PrincipalConfiguration).where(
            PrincipalConfiguration.owner_id == principal_id
        )

        result = await session.execute(query)
        # Fetch all records from the result
        config_records = result.scalars().all()

        # Extract the configuration dictionary from each record
        return [config_records.to_json() for config_records in config_records]

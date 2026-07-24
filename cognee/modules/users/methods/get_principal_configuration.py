from sqlalchemy import select
from uuid import UUID
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.PrincipalConfiguration import PrincipalConfiguration


async def get_principal_configuration(config_id: UUID, principal_id: UUID) -> dict:
    """
    Retrieves a specific Cognee configuration by identifier and owner.

    Args:
        config_id (UUID): The unique identifier of the config.
        principal_id (UUID): The unique identifier of the config owner.

    Returns:
        dict: The configuration data if found, or an empty dictionary if not found.
    """
    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        query = select(PrincipalConfiguration).where(
            PrincipalConfiguration.id == config_id,
            PrincipalConfiguration.owner_id == principal_id,
        )

        result = await session.execute(query)
        config_record = result.scalars().first()

        # Return the configuration dictionary if the record exists, otherwise an empty dict
        return config_record.configuration if config_record else {}


async def get_principal_all_configuration(principal_id: UUID) -> list[dict[str, dict]]:
    """
    Retrieves all Cognee configurations for a specific principal.

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

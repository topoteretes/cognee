from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.PrincipalConfiguration import PrincipalConfiguration


async def get_principal_configuration(principal_id: str, name: str) -> dict:
    """
    Retrieves a specific Cognee configuration for a principal by its name.

    Args:
        principal_id (str): The unique identifier of the owner (user/group).
        name (str): The specific name of the configuration to retrieve.

    Returns:
        dict: The configuration data if found, or an empty dictionary (or None) if not found.
    """
    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        query = select(PrincipalConfiguration).where(
            PrincipalConfiguration.owner_id == principal_id,
            PrincipalConfiguration.name == name,
        )

        result = await session.execute(query)
        config_record = result.scalars().first()

        # Return the configuration dictionary if the record exists, otherwise an empty dict
        return config_record.configuration if config_record else {}

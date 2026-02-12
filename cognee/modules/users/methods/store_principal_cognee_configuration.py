from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.PrincipalCogneeConfiguration import PrincipalCogneeConfiguration


async def store_principal_cognee_configuration(
    principal_id: str, name: str, configuration: dict
) -> PrincipalCogneeConfiguration:
    """
    Persists or updates a Cognee configuration for a specific principal (user or group).

    This function manages the lifecycle of principal-specific settings, such as
    Knowledge Graph schemas, LLM system prompts, and ingestion parameters. If a
    configuration with the same name already exists for the principal, it updates
    the existing record.

    Args:
        principal_id (str): The unique identifier (UUID) of the owner (user/group).
        name (str): A descriptive name for the configuration (e.g., "default_llm_settings").
        configuration (dict): A dictionary containing the JSON-serializable config data.

    Returns:
        PrincipalCogneeConfiguration: The created or updated database model instance.
    """
    relational_engine = await get_relational_engine()

    async with relational_engine.get_async_session() as session:
        # Check if a configuration with this name already exists for the principal
        query = select(PrincipalCogneeConfiguration).where(
            PrincipalCogneeConfiguration.owner_id == principal_id,
            PrincipalCogneeConfiguration.name == name,
        )
        result = await session.execute(query)
        existing_config = result.scalars().first()

        if existing_config:
            # Update existing configuration
            existing_config.configuration = configuration
            # updated_at is handled by the 'onupdate' trigger in the model
            config_record = existing_config
        else:
            # Create new configuration record
            config_record = PrincipalCogneeConfiguration(
                owner_id=principal_id, name=name, configuration=configuration
            )
            session.add(config_record)

        await session.commit()
        await session.refresh(config_record)

        return config_record

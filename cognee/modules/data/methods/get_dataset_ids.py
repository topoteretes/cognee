from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.exceptions import (
    AmbiguousDatasetNameError,
    DatasetNotFoundError,
    DatasetTypeError,
)
from cognee.modules.data.methods import get_datasets
from cognee.modules.users.methods.get_agent_user_ids import get_agent_user_ids


async def get_dataset_ids(datasets: list[str] | list[UUID], user, strict: bool = False):
    """
    Function returns dataset IDs necessary based on provided input.
    It transforms raw strings into real dataset_ids with keeping write permissions in mind.
    If a user wants to write to a dataset he is not the owner of it must be provided through UUID.
    Args:
        datasets:
        user:
        strict: When True, a name that resolves to no dataset the user or their
            agents own in the current tenant raises DatasetNotFoundError naming the misses, instead of
            being dropped. Read paths use this; write paths keep the default so an
            unmatched name can still mean "create it".

    Returns: a list of write access dataset_ids if they exist

    """
    if all(isinstance(dataset, UUID) for dataset in datasets):
        # Return list of dataset UUIDs
        dataset_ids = datasets
    else:
        # Convert list of dataset names to dataset UUID
        if all(isinstance(dataset, str) for dataset in datasets):
            # A name resolves among the datasets the user owns in the current
            # tenant, then among their agents' (a dataset not owned by either must
            # be given by id). The user's own dataset wins a name both use.
            matched = await _owned_datasets_named(user.id, datasets, user.tenant_id)
            unresolved = [name for name in dict.fromkeys(datasets) if name not in _names(matched)]
            if unresolved:
                matched.extend(await _agent_datasets_named(user, unresolved))
            if strict:
                resolved_names = {dataset.name for dataset in matched}
                missing = [name for name in dict.fromkeys(datasets) if name not in resolved_names]
                if missing:
                    raise DatasetNotFoundError(
                        message=f"Dataset(s) not found: {', '.join(repr(name) for name in missing)}. "
                        "Dataset names resolve only among the datasets you and your agents own "
                        "in the current tenant; pass dataset_ids for datasets shared with you."
                    )
            dataset_ids = [dataset.id for dataset in matched]
        else:
            raise DatasetTypeError(
                f"One or more of the provided dataset types is not handled: {datasets}"
            )

    return dataset_ids


def _names(datasets) -> set[str]:
    return {dataset.name for dataset in datasets}


async def _owned_datasets_named(owner_id: UUID, names: list[str], tenant_id) -> list:
    return [
        dataset
        for dataset in await get_datasets(owner_id)
        if dataset.name in names and dataset.tenant_id == tenant_id
    ]


async def _agent_datasets_named(user, names: list[str]) -> list:
    """One dataset per name among the user's agents' datasets; several raise."""
    async with get_relational_engine().get_async_session() as session:
        agent_ids = await get_agent_user_ids(session, user.id)
    candidates = []
    for agent_id in agent_ids:
        candidates.extend(await _owned_datasets_named(agent_id, names, user.tenant_id))
    resolved = []
    for name in names:
        named = [dataset for dataset in candidates if dataset.name == name]
        if len(named) > 1:
            raise AmbiguousDatasetNameError(name, [dataset.id for dataset in named])
        resolved.extend(named)
    return resolved

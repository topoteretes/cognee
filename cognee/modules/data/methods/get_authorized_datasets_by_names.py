from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods.get_authorized_existing_datasets import (
    get_authorized_existing_datasets,
)
from cognee.modules.data.models import Dataset
from cognee.modules.users.models import User


async def get_authorized_datasets_by_names(
    dataset_names: list[str], permission_type: str, user: User
) -> list[Dataset]:
    """Resolve every requested dataset name, or fail the request naming the misses.

    Name resolution is a membership filter (``get_dataset_ids``): a name that
    matches nothing simply drops out of the result. For a search that is silent
    data loss — the caller gets a 200 carrying its *other* datasets' answers and
    no indication that one of the datasets it asked for was never searched. The
    ``dataset_ids`` path has always failed the whole request in that situation
    (``get_specific_user_permission_datasets`` raises ``PermissionDeniedError``
    when it resolves fewer datasets than were requested); this gives the names
    path the same all-or-nothing contract.

    The failure is a 404 rather than the id path's 403 because the two misses
    mean different things: an unknown id may belong to another tenant, so it
    can only be answered with "denied", while names are resolved solely among
    the datasets this user owns in their own tenant — a miss there describes
    nothing but the caller's own namespace, so it can be reported precisely.
    It also keeps this path self-consistent: a request in which *every* name
    missed already raised ``DatasetNotFoundError``.

    An empty ``dataset_names`` keeps its "every dataset the user can read"
    meaning and never raises here.
    """
    datasets = await get_authorized_existing_datasets(dataset_names, permission_type, user)

    resolved_names = {dataset.name for dataset in datasets}
    # dict.fromkeys: report each name once, in the order it was requested.
    missing = [name for name in dict.fromkeys(dataset_names) if name not in resolved_names]

    if missing:
        raise DatasetNotFoundError(
            message=(
                f"Dataset(s) not found: {', '.join(repr(name) for name in missing)}. "
                "Dataset names resolve only among the datasets you own in the current "
                "tenant — pass dataset_ids for datasets shared with you."
            )
        )

    return datasets

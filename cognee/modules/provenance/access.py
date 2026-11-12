"""Who may read the ledger: one rule for every read surface.

Every ledger key names its dataset, so a read follows that dataset's ACL. The
HTTP routes and the CLI both run these checks with the acting user. The
manager underneath is the storage-level API, like a graph or vector engine,
and applies none: anything that reads the ledger for a user goes through here.
"""

from typing import Any
from uuid import UUID

from cognee.modules.data.methods import get_authorized_dataset
from cognee.modules.users.exceptions import PermissionDeniedError

from .tombstones import dataset_id_from_ledger_key


def _denied(message: str) -> PermissionDeniedError:
    return PermissionDeniedError(message=message, log=False)


def _require_superuser(user: Any, message: str) -> None:
    if not getattr(user, "is_superuser", False):
        raise _denied(message)


async def _require_dataset_read(user: Any, dataset_id: UUID) -> None:
    """Read access to the dataset, or a superuser.

    A superuser may already read the whole ledger, so one dataset's rows are
    theirs to read too, and that holds after the dataset is deleted: its rows
    and tombstones stay in the ledger, while the dataset and its permissions,
    which the ACL lookup needs, are gone.
    """
    if getattr(user, "is_superuser", False):
        return
    if await get_authorized_dataset(user, dataset_id, "read") is None:
        raise _denied(f"No read access to dataset {dataset_id}.")


async def authorize_ledger_scope(user: Any, dataset_id: UUID | None) -> None:
    """A dataset's rows need read access to it; the whole ledger is superuser-only."""
    if dataset_id is None:
        _require_superuser(
            user, "A dataset is required; the whole ledger is readable by superusers only."
        )
        return
    await _require_dataset_read(user, dataset_id)


async def authorize_ledger_key(user: Any, entity_id: str) -> None:
    """One row needs read access to the dataset its key names.

    A key without a dataset prefix (a custom pipeline that bypassed the
    writer's namespacing) belongs to no dataset, so it is superuser-only.
    """
    dataset_id = dataset_id_from_ledger_key(entity_id)
    if dataset_id is None:
        _require_superuser(user, "Unscoped ledger keys are readable by superusers only.")
        return
    await _require_dataset_read(user, dataset_id)


async def authorize_anchoring(user: Any) -> None:
    """Anchoring signs and checks the head of the whole chain, so it is superuser-only."""
    _require_superuser(user, "Anchoring is superuser-only.")

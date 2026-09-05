"""Explicit, permission-checked promotion of one persisted memory snapshot."""

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.storage import get_file_storage
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset
from cognee.modules.data.models import Data
from cognee.modules.users.methods import get_user
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods.get_principal import get_principal
from cognee.modules.users.permissions.methods.get_principal_datasets import get_principal_datasets
from cognee.shared.logging_utils import get_logger


@dataclass(frozen=True)
class PromotionResult:
    source_data_id: UUID
    target_data_id: UUID
    source_dataset_id: UUID
    target_dataset_id: UUID
    level: str
    status: Literal["planned", "copied", "already_promoted"]


async def _get_data(data_id: UUID, dataset_id: UUID) -> Data | None:
    async with get_relational_engine().get_async_session() as session:
        return (
            await session.execute(
                select(Data).where(Data.id == data_id, Data.dataset_id == dataset_id)
            )
        ).scalar_one_or_none()


async def _persist_copy(row, content, provenance, actor, target, target_id) -> bool:
    """Insert only: a concurrent retry must never UPDATE an existing memory."""
    storage = get_file_storage(get_base_config().data_root_directory)
    extension = str(row.extension or "txt").lstrip(".")
    if not extension.isalnum():
        extension = "txt"
    # Each attempt gets its own object. The DB primary key selects the winner;
    # concurrent writers never overwrite each other's storage or user edits.
    relative = f"{actor.id}/promotions/{target.id}/{uuid4().hex}.{extension}"
    location = await storage.store(relative, io.BytesIO(content))
    copied = Data(
        id=target_id,
        dataset_id=target.id,
        owner_id=actor.id,
        tenant_id=target.tenant_id,
        name=row.name,
        label=row.label,
        extension=extension,
        mime_type=row.mime_type,
        original_extension=extension,
        original_mime_type=row.mime_type,
        loader_engine=row.loader_engine,
        raw_data_location=location,
        original_data_location=location,
        content_hash=row.raw_content_hash,
        raw_content_hash=row.raw_content_hash,
        data_size=len(content),
        token_count=-1,
        external_metadata=row.external_metadata,
        system_metadata={"promotion": provenance},
        pipeline_status={},
    )
    try:
        async with get_relational_engine().get_async_session() as session:
            session.add(copied)
            await session.commit()
        return True
    except Exception as error:
        try:
            await storage.remove(relative)
        except Exception:  # noqa: BLE001 - cleanup must preserve the original persistence error
            get_logger("promotion").exception("Could not clean an uncommitted promotion object")
        if isinstance(error, IntegrityError):
            winner = await _get_data(target_id, target.id)
            if (
                winner is not None
                and (winner.system_metadata or {}).get("promotion", {}).get("source_revision")
                == provenance["source_revision"]
            ):
                return False
        raise


async def promote(
    data_id: UUID,
    *,
    source_dataset_id: UUID,
    target_dataset_id: UUID,
    level: Literal["user", "team"],
    reason: str,
    user: User,
    dry_run: bool = False,
    max_bytes: int = 64 * 1024 * 1024,
) -> PromotionResult:
    """Copy selected persisted memory upward, without changing source ACLs.

    ``level='user'`` permits an agent-owned dataset -> its parent's dataset.
    ``level='team'`` permits a human-owned dataset -> a dataset already shared
    for reading with the current tenant. The caller needs source read AND
    share permissions, and target write permission; ancestry is not authority.

    Agents choose which persisted document to promote and why. Inspect the
    selected document: a persisted session window can contain several entries. This operation never grants itself
    permissions, impersonates the target owner, copies other memories, or runs
    an LLM. Session entries must first be persisted with improve(). The copied
    document is ingest-complete; call cognify(datasets=[target_dataset_id],
    user=user) explicitly to make it searchable in the destination graph.

    Retries of the same source revision and destination return the same data
    ID without overwriting destination edits. Changed source content produces a
    new snapshot. Source edits/deletion do not propagate to promoted copies.
    ``dry_run`` performs authorization and relationship checks without writes.
    """
    if level not in ("user", "team") or not reason.strip() or max_bytes <= 0:
        raise ValueError("A user/team level, nonempty reason and positive max_bytes are required")
    data_id, source_dataset_id, target_dataset_id = map(
        UUID, map(str, (data_id, source_dataset_id, target_dataset_id))
    )
    if source_dataset_id == target_dataset_id:
        raise ValueError("Promotion requires different source and target datasets")
    if user is None:
        raise ValueError("Promotion requires an explicit authenticated user")
    actor = await get_user(user.id)
    source = await get_authorized_dataset(actor, source_dataset_id, "read")
    shareable = await get_authorized_dataset(actor, source_dataset_id, "share")
    target = await get_authorized_dataset(actor, target_dataset_id, "write")
    if source is None or target is None or shareable is None:
        raise ValueError("Source or target dataset is unavailable")
    source_owner = await get_user(source.owner_id)
    if level == "user":
        if source_owner.parent_user_id != target.owner_id:
            raise ValueError("User promotion must target the source agent's parent")
    else:
        if source_owner.parent_user_id is not None:
            raise ValueError("Promote agent memory to its user before promoting to a team")
        if not actor.tenant_id or target.tenant_id != actor.tenant_id:
            raise ValueError("Team promotion must stay in the caller's current tenant")
        tenant = await get_principal(actor.tenant_id)
        team_datasets = await get_principal_datasets(tenant, "read")
        if target.id not in {dataset.id for dataset in team_datasets}:
            raise ValueError("Target is not shared for reading with this team")
    row = await _get_data(data_id, source_dataset_id)
    if row is None:
        raise ValueError("Selected memory is unavailable in the source dataset")
    stored_digest = row.raw_content_hash
    if not stored_digest:
        raise ValueError("Source memory lacks a content revision; re-ingest before promotion")
    async with open_data_file(row.raw_data_location, "rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("Selected memory exceeds max_bytes")
    if hashlib.md5(content).hexdigest() != stored_digest:
        raise ValueError("Source changed during promotion; retry with the current revision")
    revision = hashlib.sha256(content).hexdigest()
    target_id = uuid5(
        NAMESPACE_URL,
        f"cognee:promotion:v1:{source_dataset_id}:{data_id}:{revision}:{target_dataset_id}",
    )
    base = {
        "source_data_id": data_id,
        "target_data_id": target_id,
        "source_dataset_id": source_dataset_id,
        "target_dataset_id": target_dataset_id,
        "level": level,
    }
    existing = await _get_data(target_id, target_dataset_id)
    if existing is not None:
        provenance = (existing.system_metadata or {}).get("promotion", {})
        if (
            provenance.get("source_data_id") != str(data_id)
            or provenance.get("source_revision") != revision
        ):
            raise ValueError("Destination ID conflicts with a different memory")
        return PromotionResult(**base, status="already_promoted")
    if dry_run:
        return PromotionResult(**base, status="planned")
    provenance = {
        "source_data_id": str(data_id),
        "source_dataset_id": str(source_dataset_id),
        "source_revision": revision,
        "target_dataset_id": str(target_dataset_id),
        "promoted_by": str(actor.id),
        "level": level,
        "reason": reason.strip(),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "previous": (row.system_metadata or {}).get("promotion"),
    }
    created = await _persist_copy(row, content, provenance, actor, target, target_id)
    return PromotionResult(**base, status="copied" if created else "already_promoted")

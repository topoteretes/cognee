import inspect
import json
import re
import warnings
from uuid import UUID
from sqlalchemy import select
from typing import Any, BinaryIO, Union

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.utils.get_file_metadata import FileMetadata
from ..models.Metadata import Metadata


async def write_metadata(
    data_item: Union[BinaryIO, str, Any], data_id: UUID, file_metadata: FileMetadata
) -> UUID:
    metadata_dict = get_metadata_dict(data_item, file_metadata)
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        metadata = (
            await session.execute(select(Metadata).filter(Metadata.data_id == data_id))
        ).scalar_one_or_none()

        if metadata is not None:
            metadata.metadata_repr = json.dumps(metadata_dict)
            metadata.metadata_source = parse_type(type(data_item))
            await session.merge(metadata)
        else:
            metadata = Metadata(
                id=data_id,
                metadata_repr=json.dumps(metadata_dict),
                metadata_source=parse_type(type(data_item)),
                data_id=data_id,
            )
            session.add(metadata)

        await session.commit()


def parse_type(type_: Any) -> str:
    pattern = r".+'([\w_\.]+)'"
    match = re.search(pattern, str(type_))
    if match:
        return match.group(1)
    else:
        raise Exception(f"type: {type_} could not be parsed")


def get_metadata_dict(
    data_item: Union[BinaryIO, str, Any], file_metadata: FileMetadata
) -> dict[str, Any]:
    if isinstance(data_item, str):
        return file_metadata
    elif isinstance(data_item, BinaryIO):
        return file_metadata
    elif hasattr(data_item, "dict") and inspect.ismethod(getattr(data_item, "dict")):
        return {**file_metadata, **data_item.dict()}
    else:
        warnings.warn(
            f"metadata of type {type(data_item)}: {str(data_item)[:20]}... does not have dict method. Defaulting to string method"
        )
        try:
            return {**dict(file_metadata), "content": str(data_item)}
        except Exception as e:
            raise Exception(f"Could not cast metadata to string: {e}")

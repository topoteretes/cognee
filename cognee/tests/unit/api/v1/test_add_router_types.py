"""
Unit tests for type annotation fixes in add/update/cognify routers.
Verifies that Optional and Union types are correctly applied (issue #2049).
"""
import inspect
import uuid
from typing import get_type_hints

import pytest


def test_add_router_data_is_optional():
    """data parameter in POST /v1/add should be Optional[List[UploadFile]]."""
    from cognee.api.v1.add.routers.get_add_router import get_add_router
    from fastapi import UploadFile
    from typing import List, Optional

    router = get_add_router()
    route = next(r for r in router.routes if r.path == "")
    endpoint = route.endpoint
    sig = inspect.signature(endpoint)
    data_param = sig.parameters["data"]

    # Should have a default of None (making it optional)
    # FastAPI wraps the default in a FieldInfo object e.g. File(None), so check .default attribute
    assert data_param.default is not inspect.Parameter.empty, (
        "data parameter should have a default value"
    )
    # FastAPI's File(None) returns a FieldInfo with default=None
    default_val = getattr(data_param.default, "default", data_param.default)
    assert default_val is None, (
        f"data parameter effective default should be None, got {default_val!r}"
    )


def test_update_router_data_is_optional():
    """data parameter in PATCH /v1/update should be Optional[List[UploadFile]]."""
    import inspect
    from cognee.api.v1.update.routers.get_update_router import get_update_router

    router = get_update_router()
    route = next(r for r in router.routes if r.path == "")
    endpoint = route.endpoint
    sig = inspect.signature(endpoint)
    data_param = sig.parameters["data"]

    assert data_param.default is not inspect.Parameter.empty, (
        "data parameter should have a default value"
    )


def test_cognify_dto_datasets_accepts_union():
    """CognifyPayloadDTO.datasets should accept Union[str, List[str]]."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO
    from typing import Union, List, Optional, get_args, get_origin
    import typing

    fields = CognifyPayloadDTO.model_fields
    assert "datasets" in fields, "CognifyPayloadDTO must have datasets field"

    # Get the annotation
    hints = get_type_hints(CognifyPayloadDTO)
    datasets_type = hints["datasets"]

    # Should be Optional (i.e., Union[..., None])
    # And the inner type should allow both str and List[str]
    args = get_args(datasets_type)
    # Flatten all args including nested Union args
    flat_args = []
    for arg in args:
        if get_origin(arg) is Union:
            flat_args.extend(get_args(arg))
        else:
            flat_args.append(arg)

    type_strs = [str(a) for a in flat_args]
    has_str = str in flat_args or any("str" in s and "List" not in s for s in type_strs)
    has_list_str = any(
        (get_origin(a) is list and str in get_args(a)) or "List[str]" in str(a)
        for a in flat_args
    )

    assert has_str or has_list_str, (
        f"datasets type {datasets_type} should support str or List[str], got args: {flat_args}"
    )


def test_cognify_dto_dataset_ids_accepts_union():
    """CognifyPayloadDTO.dataset_ids should accept Union[UUID, List[UUID]]."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO
    from typing import get_args, get_origin, Union
    from uuid import UUID

    hints = get_type_hints(CognifyPayloadDTO)
    dataset_ids_type = hints["dataset_ids"]

    args = get_args(dataset_ids_type)
    flat_args = []
    for arg in args:
        if get_origin(arg) is Union:
            flat_args.extend(get_args(arg))
        else:
            flat_args.append(arg)

    has_uuid = UUID in flat_args or any("UUID" in str(a) and "List" not in str(a) for a in flat_args)
    has_list_uuid = any(
        (get_origin(a) is list and UUID in get_args(a)) or "List[uuid.UUID]" in str(a)
        for a in flat_args
    )

    assert has_uuid or has_list_uuid, (
        f"dataset_ids type {dataset_ids_type} should support UUID or List[UUID], got args: {flat_args}"
    )


def test_cognify_dto_datasets_none_default():
    """CognifyPayloadDTO.datasets default should be None."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    instance = CognifyPayloadDTO()
    assert instance.datasets is None


def test_cognify_dto_accepts_single_string_dataset():
    """CognifyPayloadDTO should accept a single string for datasets."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    dto = CognifyPayloadDTO(datasets="my_dataset")
    assert dto.datasets == "my_dataset"


def test_cognify_dto_accepts_list_string_dataset():
    """CognifyPayloadDTO should accept a list of strings for datasets."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    dto = CognifyPayloadDTO(datasets=["ds1", "ds2"])
    assert dto.datasets == ["ds1", "ds2"]


def test_cognify_dto_accepts_single_uuid_dataset_id():
    """CognifyPayloadDTO should accept a single UUID for dataset_ids."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    single_id = uuid.uuid4()
    dto = CognifyPayloadDTO(dataset_ids=single_id)
    assert dto.dataset_ids == single_id


def test_cognify_dto_accepts_list_uuid_dataset_ids():
    """CognifyPayloadDTO should accept a list of UUIDs for dataset_ids."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    ids = [uuid.uuid4(), uuid.uuid4()]
    dto = CognifyPayloadDTO(dataset_ids=ids)
    assert dto.dataset_ids == ids

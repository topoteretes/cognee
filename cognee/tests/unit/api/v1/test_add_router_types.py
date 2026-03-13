"""
Unit tests for type annotation fixes in add/update/cognify routers.
Verifies that Optional and Union types are correctly applied (issue #2049).
"""
import inspect
import uuid
from typing import Union, get_args, get_origin, get_type_hints, List, Optional

import pytest

def flatten_union_args(type_hint) -> list:
    """Helper to flatten nested Union types into a list of args."""
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if origin is Union:
        flat = []
        for arg in args:
            flat.extend(flatten_union_args(arg))
        return flat
    return [type_hint]

def test_add_router_data_is_optional() -> None:
    """data parameter in POST /v1/add should be Optional[List[UploadFile]] with default None."""
    from cognee.api.v1.add.routers.get_add_router import get_add_router

    router = get_add_router()
    route = next(
        r for r in router.routes if r.path == "" and "POST" in getattr(r, "methods", set())
    )
    endpoint = route.endpoint
    
    # Check type hint
    hints = get_type_hints(endpoint)
    data_type = hints["data"]
    data_args = flatten_union_args(data_type)
    
    assert type(None) in data_args, f"data annotation should be Optional, got {data_type!r}"
    assert any(get_origin(a) is list for a in data_args), f"data annotation should include list, got {data_type!r}"

    # Check default value
    sig = inspect.signature(endpoint)
    data_param = sig.parameters["data"]
    assert data_param.default is not inspect.Parameter.empty, "data parameter should have a default value"
    default_val = getattr(data_param.default, "default", data_param.default)
    assert default_val is None, f"data parameter effective default should be None, got {default_val!r}"


def test_update_router_data_is_optional() -> None:
    """data parameter in PATCH /v1/update should be Optional[List[UploadFile]] with default None."""
    from cognee.api.v1.update.routers.get_update_router import get_update_router

    router = get_update_router()
    route = next(
        r for r in router.routes if r.path == "" and "PATCH" in getattr(r, "methods", set())
    )
    endpoint = route.endpoint
    
    # Check type hint
    hints = get_type_hints(endpoint)
    data_type = hints["data"]
    data_args = flatten_union_args(data_type)
    
    assert type(None) in data_args, f"data annotation should be Optional, got {data_type!r}"
    assert any(get_origin(a) is list for a in data_args), f"data annotation should include list, got {data_type!r}"

    # Check default value
    sig = inspect.signature(endpoint)
    data_param = sig.parameters["data"]
    assert data_param.default is not inspect.Parameter.empty, "data parameter should have a default value"
    default_val = getattr(data_param.default, "default", data_param.default)
    assert default_val is None, f"data parameter effective default should be None, got {default_val!r}"


def test_cognify_dto_datasets_accepts_union() -> None:
    """CognifyPayloadDTO.datasets should accept both str and List[str] and None."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    hints = get_type_hints(CognifyPayloadDTO)
    datasets_type = hints["datasets"]
    flat_args = flatten_union_args(datasets_type)

    has_str = str in flat_args or any("str" in str(a) and "List" not in str(a) for a in flat_args)
    has_list_str = any((get_origin(a) is list and str in get_args(a)) or "List[str]" in str(a) for a in flat_args)
    has_none = type(None) in flat_args

    assert has_str and has_list_str and has_none, (
        f"datasets type {datasets_type} must support str, List[str], and None, got args: {flat_args}"
    )


def test_cognify_dto_dataset_ids_accepts_union() -> None:
    """CognifyPayloadDTO.dataset_ids should accept both UUID and List[UUID] and None."""
    from uuid import UUID
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    hints = get_type_hints(CognifyPayloadDTO)
    dataset_ids_type = hints["dataset_ids"]
    flat_args = flatten_union_args(dataset_ids_type)

    has_uuid = UUID in flat_args or any("UUID" in str(a) and "List" not in str(a) for a in flat_args)
    has_list_uuid = any((get_origin(a) is list and UUID in get_args(a)) or "List[uuid.UUID]" in str(a) for a in flat_args)
    has_none = type(None) in flat_args

    assert has_uuid and has_list_uuid and has_none, (
        f"dataset_ids type {dataset_ids_type} must support UUID, List[UUID], and None, got args: {flat_args}"
    )


def test_cognify_dto_datasets_none_default() -> None:
    """CognifyPayloadDTO.datasets default should be None."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    instance = CognifyPayloadDTO()
    assert instance.datasets is None


def test_cognify_dto_accepts_single_string_dataset() -> None:
    """CognifyPayloadDTO should accept a single string for datasets."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    dto = CognifyPayloadDTO(datasets="my_dataset")
    assert dto.datasets == "my_dataset"


def test_cognify_dto_accepts_list_string_dataset() -> None:
    """CognifyPayloadDTO should accept a list of strings for datasets."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    dto = CognifyPayloadDTO(datasets=["ds1", "ds2"])
    assert dto.datasets == ["ds1", "ds2"]


def test_cognify_dto_accepts_single_uuid_dataset_id() -> None:
    """CognifyPayloadDTO should accept a single UUID for dataset_ids."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    single_id = uuid.uuid4()
    dto = CognifyPayloadDTO(dataset_ids=single_id)
    assert dto.dataset_ids == single_id


def test_cognify_dto_accepts_list_uuid_dataset_ids() -> None:
    """CognifyPayloadDTO should accept a list of UUIDs for dataset_ids."""
    from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO

    ids = [uuid.uuid4(), uuid.uuid4()]
    dto = CognifyPayloadDTO(dataset_ids=ids)
    assert dto.dataset_ids == ids

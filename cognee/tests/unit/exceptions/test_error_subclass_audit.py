import inspect

import pytest

from cognee.exceptions import (
    CogneeApiError,
    CogneeConfigurationError,
    CogneeDataNotReadyError,
    CogneePermissionError,
    CogneeSystemError,
    CogneeTransientError,
    CogneeValidationError,
    SEMANTIC_ERROR_BASES,
    SPECIAL_CASE_OVERRIDES,
)


def _all_cognee_api_error_subclasses() -> list[type[CogneeApiError]]:
    subclasses: list[type[CogneeApiError]] = []
    for module in (
        "cognee.modules.retrieval.exceptions.exceptions",
        "cognee.modules.data.exceptions.exceptions",
        "cognee.modules.users.exceptions.exceptions",
        "cognee.infrastructure.databases.exceptions.exceptions",
        "cognee.infrastructure.llm.exceptions",
    ):
        imported = __import__(module, fromlist=["*"])
        for _, obj in inspect.getmembers(imported, inspect.isclass):
            if issubclass(obj, CogneeApiError) and obj is not CogneeApiError:
                subclasses.append(obj)
    return subclasses


def _has_semantic_base(cls: type[CogneeApiError]) -> bool:
    if cls in SEMANTIC_ERROR_BASES:
        return True
    for base in cls.__mro__[1:]:
        if base in SEMANTIC_ERROR_BASES:
            return True
    return False


def test_special_case_overrides_are_documented():
    for exc_type in SPECIAL_CASE_OVERRIDES:
        assert issubclass(exc_type, CogneeApiError)


def test_pillar_b_targets_use_semantic_bases():
    from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError, UnauthorizedDataAccessError
    from cognee.modules.retrieval.exceptions.exceptions import NoDataError
    from cognee.modules.users.exceptions.exceptions import PermissionDeniedError

    assert issubclass(NoDataError, CogneeDataNotReadyError)
    assert issubclass(DatasetNotFoundError, CogneeDataNotReadyError)
    assert issubclass(UnauthorizedDataAccessError, CogneePermissionError)
    assert issubclass(PermissionDeniedError, CogneePermissionError)


def test_category_bases_set_default_codes():
    assert CogneeValidationError.default_code.value == "invalid_input"
    assert CogneeConfigurationError.default_code.value == "missing_config"
    assert CogneeDataNotReadyError.default_code.value == "data_not_ready"
    assert CogneePermissionError.default_code.value == "permission_denied"
    assert CogneeTransientError.default_code.value == "transient"
    assert CogneeSystemError.default_code.value == "system"

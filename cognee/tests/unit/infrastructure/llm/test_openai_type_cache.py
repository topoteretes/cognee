"""Unit tests for the openai type-introspection cache (cognee.infrastructure.llm.openai_type_cache)."""

from typing import Annotated, List, Literal, Optional, Union

import pytest

pytest.importorskip("openai")

# Importing the llm package installs the cache as an import side effect.
from cognee.infrastructure.llm import openai_type_cache
from openai import _models
from openai._utils import _compat as ou_compat
from openai._utils import _typing as ou_typing


def test_cache_is_installed_on_package_import_and_install_is_idempotent():
    assert openai_type_cache._INSTALLED is True
    # A second install must be a no-op.
    assert openai_type_cache.install() is False


def test_helpers_are_rebound_to_shared_cached_wrappers():
    # The same cached wrapper must be rebound at every import site, and
    # functools.wraps exposes the uncached original via __wrapped__.
    assert hasattr(ou_compat.get_origin, "__wrapped__")
    assert hasattr(ou_compat.get_args, "__wrapped__")
    assert hasattr(ou_compat.is_literal_type, "__wrapped__")
    assert hasattr(ou_typing.is_annotated_type, "__wrapped__")

    assert _models.get_origin is ou_compat.get_origin
    assert _models.get_args is ou_compat.get_args
    assert _models.is_literal_type is ou_compat.is_literal_type
    assert _models.is_annotated_type is ou_typing.is_annotated_type


def test_cached_helpers_match_uncached_originals():
    type_cases = [
        int,
        list[int],
        List[int],
        dict[str, int],
        Optional[str],
        Union[int, str],
        Literal["a", "b"],
        Annotated[int, "meta"],
    ]
    for tp in type_cases:
        # Call twice so the second hit comes from the cache.
        for _ in range(2):
            assert ou_compat.get_origin(tp) == ou_compat.get_origin.__wrapped__(tp)
            assert ou_compat.get_args(tp) == ou_compat.get_args.__wrapped__(tp)
            assert ou_compat.is_literal_type(tp) == ou_compat.is_literal_type.__wrapped__(tp)
            assert ou_typing.is_annotated_type(tp) == ou_typing.is_annotated_type.__wrapped__(tp)


def test_unhashable_argument_falls_back_to_uncached_call():
    # Lists are unhashable, so lru_cache raises TypeError internally; the
    # wrapper must swallow it and return the uncached result instead.
    assert ou_compat.get_origin(["not", "a", "type"]) is None
    assert ou_compat.get_args(["not", "a", "type"]) == ()

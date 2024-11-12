import sys
import warnings
from unittest.mock import patch

import pytest

from cognee.modules.users.methods.get_default_user import get_default_user
from cognee.tests.unit.utils.get_mock_user import get_mock_user


def apply_with_to_item(module_items, fn, fn_str):
    for i, (name, module) in enumerate(module_items):
        if hasattr(module, fn_str) and not "test" in name:
            with patch(name, fn):
                if len(module_items[(i + 1) :]) > 0:
                    apply_with_to_item(module_items[(i + 1) :], fn, fn_str)


@pytest.fixture(autouse=True, scope="session")
def set_get_mock_user_wrapper():

    def get_mock_user_wrapper():
        return get_mock_user(None, None)

    module_items = list(sys.modules.items())
    apply_with_to_item(module_items, get_mock_user_wrapper(), "get_default_user")

    yield

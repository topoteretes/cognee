from unittest.mock import patch
import warnings
import pytest

from cognee.modules.users.methods.get_default_user import get_default_user
from cognee.tests.unit.utils.get_mock_user import get_mock_user
import sys


@pytest.fixture(autouse=True, scope="session")
def set_get_mock_user_wrapper():

    def get_mock_user_wrapper():
        return get_mock_user(None, None)
    
    for name, module in sys.modules.items():
        if hasattr(module, 'get_default_user'):
            warnings.warn(f"Found get_default_user in module: {name}")
    
    with patch(
        "cognee.modules.users.methods.get_default_user", get_mock_user_wrapper()
    ):
        with patch(
            "cognee.modules.users.methods", get_mock_user_wrapper()
        ):
            with patch('cognee.modules.pipelines.operations.run_tasks', get_mock_user_wrapper()):
                yield

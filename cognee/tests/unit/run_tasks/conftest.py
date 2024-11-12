from unittest.mock import patch

import pytest

from cognee.modules.users.methods.get_default_user import get_default_user
from cognee.tests.unit.utils.get_mock_user import get_mock_user


@pytest.fixture(autouse=True, scope="session")
def set_get_mock_user_wrapper():

    def get_mock_user_wrapper():
        return get_mock_user(None, None)

    with patch(
        "cognee.modules.users.methods.get_default_user", get_mock_user_wrapper()
    ):
        yield

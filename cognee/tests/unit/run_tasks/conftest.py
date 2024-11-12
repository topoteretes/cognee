import pytest
import warnings
from cognee.tests.unit.utils.get_mock_user import get_mock_user
from cognee.modules.users.methods.get_default_user import get_default_user
import cognee


@pytest.fixture(autouse=True, scope="session")
def set_get_mock_user_wrapper():

    def get_mock_user_wrapper():
        warnings.warn("\n\n\n---------------USING MOCK USER--------------------\n\n\n")
        return get_mock_user(None, None)

    get_default_user = cognee.modules.users.methods.get_default_user
    cognee.modules.users.methods.get_default_user = get_mock_user_wrapper()
    yield
    cognee.modules.users.methods.get_default_user = get_default_user

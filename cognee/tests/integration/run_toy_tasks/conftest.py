
import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def copy_cognee_db_to_target_location():

    os.system
    yield

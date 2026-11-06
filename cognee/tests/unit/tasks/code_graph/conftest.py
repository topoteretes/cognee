import pytest


@pytest.fixture(autouse=True)
def allow_http_requests(monkeypatch):
    """Remote cloning is opt-in (``ALLOW_HTTP_REQUESTS`` is off by default).

    The tests here exercise cloning with git mocked out, so they opt in once.
    A test of the refusal itself unsets or overrides the variable in its body,
    which runs after this fixture.
    """
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")

"""A revoked connection ends every connector's run with the revocation, never a clean cut.

A connector stops on its own when its request budget is spent, and that stop sends no
request, so the host is not asked through the client. ``dlt_utils.guarded_rows`` asks it
before every row and once more when the rows end, so the extraction fails and dlt loads
nothing; a connector without it reports a revoked run as a budget cut and its rows load.
One case per connector with a request budget, each at the smallest budget it accepts; a
new one fails ``test_every_connector_with_a_request_budget_is_covered`` until it has a
case here.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Callable

import pytest

from cognee.tasks.ingestion import connectors as connectors_package
from cognee.tasks.ingestion.connectors.apollo import apollo_source
from cognee.tasks.ingestion.connectors.asana import asana_source
from cognee.tasks.ingestion.connectors.linear import linear_source
from cognee.tasks.ingestion.connectors.onenote import onenote_source
from cognee.tasks.ingestion.connectors.slack import slack_source
from cognee.tests.unit.tasks.connectors.fake_apollo import FakeApollo
from cognee.tests.unit.tasks.connectors.fake_asana import FakeAsana
from cognee.tests.unit.tasks.connectors.fake_onenote import Graph
from cognee.tests.unit.tasks.connectors.fake_slack import CHANNEL, TEAM, FakeSlack
from cognee.tests.unit.tasks.connectors.test_linear_source import TEAM as LINEAR_TEAM
from cognee.tests.unit.tasks.connectors.test_linear_source import FakeLinear


class _Revoked(Exception):
    pass


def _revoked_once_answered(requests: list) -> Callable[[], None]:
    """A host check that fails from the moment the provider has answered one request."""

    def check_active() -> None:
        if requests:
            raise _Revoked("access revoked")

    return check_active


def _slack(check_active_for):
    fake = FakeSlack()
    fake.post("1793872800.000100", "U1", "Where is the launch checklist?")
    fake.post("1793872860.000100", "U1", "Deploy is done.")
    return slack_source(
        service=fake,
        team_id=TEAM,
        channel_id=CHANNEL,
        resource_name="slack_revocation_test",
        check_active=check_active_for(fake.calls),
        max_requests=1,
    )


def _linear(check_active_for):
    fake = FakeLinear()
    fake.add_issue("i1", 1)
    fake.add_issue("i2", 2)
    return linear_source(
        team_id=LINEAR_TEAM,
        service=fake,
        check_active=check_active_for(fake.calls),
        max_requests=1,
    )


def _apollo(check_active_for):
    fake = FakeApollo()
    fake.add_contact("c1", "Ana Lee")
    fake.add_contact("c2", "Ben Ortiz")
    return apollo_source(
        service=fake,
        resource_name="apollo_revocation_test",
        check_active=check_active_for(fake.calls),
        max_requests=1,
    )


def _asana(check_active_for):
    fake = FakeAsana()
    fake.add_project("P1")
    fake.add_task("1", "Task 1", notes="Body 1.")
    fake.add_task("2", "Task 2", notes="Body 2.")
    return asana_source(
        service=fake,
        resource_name="asana_revocation_test",
        project_gids=["P1"],
        check_active=check_active_for(fake.calls),
        max_requests=3,  # the requests that read one task
    )


def _onenote(check_active_for):
    graph = Graph()
    return onenote_source(
        graph.token,
        notebook_id="a",
        account_id="tenant:user",
        http_client=graph.client().__enter__(),
        check_active=check_active_for(graph.requests),
        max_requests=1,
    )


BUDGETED_SOURCES = {
    "slack_source": _slack,
    "linear_source": _linear,
    "apollo_source": _apollo,
    "asana_source": _asana,
    "onenote_source": _onenote,
}


def _causes(error: BaseException):
    while error is not None:
        yield error
        error = error.__cause__ or error.__context__


@pytest.mark.parametrize("name", sorted(BUDGETED_SOURCES))
def test_a_budget_cut_never_hides_a_revocation(name):
    """Revoked while the first request is answered, then out of budget: the extraction fails."""
    resource = BUDGETED_SOURCES[name](_revoked_once_answered)

    with pytest.raises(Exception) as caught:
        list(resource)

    assert any(isinstance(cause, _Revoked) for cause in _causes(caught.value))


def test_every_connector_with_a_request_budget_is_covered():
    budgeted = set()
    for module_info in pkgutil.iter_modules(connectors_package.__path__):
        module = importlib.import_module(f"{connectors_package.__name__}.{module_info.name}")
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if (
                name.endswith("_source")
                and function.__module__ == module.__name__
                and "max_requests" in inspect.signature(function).parameters
            ):
                budgeted.add(name)

    assert budgeted == set(BUDGETED_SOURCES)

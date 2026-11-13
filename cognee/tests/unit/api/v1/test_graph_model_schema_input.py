"""cognify() and remember() take a graph model as a class or as its JSON schema dict.

The REST routes used to convert the schema themselves and pass the SDK a class. Now they
forward the dict, and ``resolve_graph_model`` is the one place a schema becomes a class:
``cognify()`` calls it for its run, ``remember()`` calls it to refuse a bad schema before
``add()`` stores anything, and hands the caller's dict on to ``cognify()`` unchanged.
"""

import importlib
import json
from types import SimpleNamespace
from typing import Annotated, get_args
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cognee.api.v1.cognify.routers.get_cognify_router import get_cognify_router
from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.exceptions import CogneeApiError, CogneeValidationError
from cognee.infrastructure.engine import DataPoint, Edge, FromIdentity
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.exceptions import (
    ExternalSchemaReferenceError,
    InvalidGraphSchemaError,
    UnresolvedSchemaReferenceError,
)
from cognee.shared.graph_model_utils import (
    EDGE_KEY,
    FROM_IDENTITY_KEY,
    graph_model_to_graph_schema,
    resolve_graph_model,
)

cognify_mod = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_mod = importlib.import_module("cognee.api.v1.remember.remember")
serve_state = importlib.import_module("cognee.api.v1.serve.state")
estimator_mod = importlib.import_module("cognee.modules.cognify.estimator")
add_pkg = importlib.import_module("cognee.api.v1.add")
cognify_pkg = importlib.import_module("cognee.api.v1.cognify")
remember_pkg = importlib.import_module("cognee.api.v1.remember")


class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Role(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Worker(DataPoint):
    name: str
    role: Annotated[Role, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class StaffGraph(DataPoint):
    people: list[Person] = []
    workers: list[Worker] = []
    knows: list[Edge[Person, Person]] = []


def _schema() -> dict:
    """A fresh copy each time: the dicts are compared by identity and mutated below."""
    return json.loads(json.dumps(graph_model_to_graph_schema(StaffGraph)))


def _without_title() -> dict:
    schema = _schema()
    schema.pop("title")
    return schema


def _external_ref() -> dict:
    return {
        "title": "G",
        "type": "object",
        "properties": {"n": {"type": "array", "items": {"$ref": "/etc/passwd"}}},
    }


def _unresolved_edge() -> dict:
    schema = _schema()
    schema["properties"]["knows"][EDGE_KEY]["target"] = "Ghost"
    return schema


def _unresolved_identity() -> dict:
    schema = _schema()
    schema["$defs"]["Worker"]["properties"]["role"][FROM_IDENTITY_KEY] = "Ghost"
    return schema


# (case, schema factory, error class, text the detail carries) — every one is a 400.
INVALID_SCHEMAS = [
    ("not_an_object", lambda: [1, 2], InvalidGraphSchemaError, "JSON schema object"),
    ("empty_object", dict, InvalidGraphSchemaError, "top-level 'title'"),
    ("no_title", _without_title, InvalidGraphSchemaError, "top-level 'title'"),
    ("external_ref", _external_ref, ExternalSchemaReferenceError, "/etc/passwd"),
    ("unresolved_edge", _unresolved_edge, UnresolvedSchemaReferenceError, "Ghost"),
    ("unresolved_identity", _unresolved_identity, UnresolvedSchemaReferenceError, "Ghost"),
]


def _list_item(annotation):
    """The class inside ``list[X]`` or ``list[X] | None``."""
    members = [arg for arg in get_args(annotation) if arg is not type(None)]
    inner = members[0] if members else annotation
    return get_args(inner)[0] if get_args(inner) else inner


def _assert_is_the_staff_graph(model):
    """The class a schema of StaffGraph builds: its title, fields and metadata default."""
    assert isinstance(model, type) and issubclass(model, DataPoint)
    assert model.__name__ == "StaffGraph"
    assert {"people", "workers", "knows"} <= set(model.model_fields)
    person = _list_item(model.model_fields["people"].annotation)
    assert person.model_fields["metadata"].default == {
        "index_fields": ["name"],
        "identity_fields": ["name"],
    }


# ----- resolve_graph_model: the one conversion site -----


def test_none_is_the_default_model():
    assert resolve_graph_model(None) is KnowledgeGraph


@pytest.mark.parametrize("model", [KnowledgeGraph, StaffGraph])
def test_a_class_is_returned_as_given(model):
    assert resolve_graph_model(model) is model


def test_a_schema_builds_the_class_it_describes():
    _assert_is_the_staff_graph(resolve_graph_model(_schema()))


@pytest.mark.parametrize(
    "case, schema, error, text", INVALID_SCHEMAS, ids=[c[0] for c in INVALID_SCHEMAS]
)
def test_an_invalid_schema_is_a_typed_400(case, schema, error, text):
    with pytest.raises(error, match=text) as raised:
        resolve_graph_model(schema())

    assert isinstance(raised.value, CogneeValidationError)
    assert raised.value.status_code == 400


def test_an_external_ref_is_refused_before_the_generator_runs():
    """The generator resolves $ref itself (URLs fetched, paths read): it must never see one."""
    with (
        patch("cognee.shared.graph_model_utils.generate") as generate,
        pytest.raises(ExternalSchemaReferenceError),
    ):
        resolve_graph_model(_external_ref())

    generate.assert_not_called()


def test_building_is_repeatable_and_leaves_the_schema_alone():
    """One schema dict builds the same model every time, and is not changed by it."""
    schema = _schema()
    snapshot = json.loads(json.dumps(schema))

    first = resolve_graph_model(schema)
    second = resolve_graph_model(schema)

    assert graph_model_to_graph_schema(first) == graph_model_to_graph_schema(second)
    assert schema == snapshot


def test_a_generator_failure_becomes_a_typed_400():
    with (
        patch("cognee.shared.graph_model_utils.generate", side_effect=RuntimeError("boom")),
        pytest.raises(InvalidGraphSchemaError, match="boom") as raised,
    ):
        resolve_graph_model(_schema())

    assert isinstance(raised.value.__cause__, RuntimeError)


# ----- cognify(): both forms give the same run -----


@pytest.fixture
def local_cognify():
    """cognify() up to its pipeline: the tasks it builds are captured, nothing runs."""
    with (
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch.object(cognify_mod, "get_pipeline_executor") as executor,
        patch.object(cognify_mod, "get_default_tasks", new_callable=AsyncMock) as tasks,
        patch(
            "cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock
        ) as migrations,
        patch.object(cognify_mod, "get_configured_ontology_resolver", return_value=None),
    ):
        tasks.return_value = []
        executor.return_value = AsyncMock(return_value={})
        yield SimpleNamespace(tasks=tasks, executor=executor, migrations=migrations)


@pytest.mark.asyncio
async def test_cognify_builds_the_tasks_from_a_schema_dict(local_cognify):
    await cognify_mod.cognify(graph_model=_schema(), extractor="llm")

    _assert_is_the_staff_graph(local_cognify.tasks.await_args.kwargs["graph_model"])


@pytest.mark.asyncio
async def test_cognify_leaves_the_callers_schema_unmodified(local_cognify):
    schema = _schema()
    snapshot = json.loads(json.dumps(schema))

    await cognify_mod.cognify(graph_model=schema, extractor="llm")

    local_cognify.tasks.assert_awaited_once()
    assert schema == snapshot


@pytest.mark.asyncio
async def test_cognify_passes_a_class_through_unchanged(local_cognify):
    await cognify_mod.cognify(graph_model=StaffGraph, extractor="llm")

    assert local_cognify.tasks.await_args.kwargs["graph_model"] is StaffGraph


@pytest.mark.asyncio
async def test_cognify_takes_none_as_the_default_model(local_cognify):
    await cognify_mod.cognify(graph_model=None, extractor="llm")

    assert local_cognify.tasks.await_args.kwargs["graph_model"] is KnowledgeGraph


@pytest.mark.asyncio
@pytest.mark.parametrize("run_in_background", [False, True])
@pytest.mark.parametrize(
    "case, schema, error, text", INVALID_SCHEMAS, ids=[c[0] for c in INVALID_SCHEMAS]
)
async def test_cognify_refuses_an_invalid_schema_before_any_run(
    local_cognify, run_in_background, case, schema, error, text
):
    with pytest.raises(error, match=text):
        await cognify_mod.cognify(graph_model=schema(), run_in_background=run_in_background)

    local_cognify.migrations.assert_not_awaited()
    local_cognify.tasks.assert_not_awaited()
    local_cognify.executor.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_attachment_accepts_a_schema_of_a_datapoint_model(local_cognify):
    await cognify_mod.cognify(graph_model=_schema(), chunk_attachment="all", extractor="llm")

    kwargs = local_cognify.tasks.await_args.kwargs
    assert kwargs["chunk_attachment"] == "all"
    _assert_is_the_staff_graph(kwargs["graph_model"])


@pytest.mark.asyncio
async def test_gliner_demo_refuses_a_schema_like_it_refuses_a_class(local_cognify):
    with (
        patch.object(cognify_mod, "ensure_extractor_runtime", new_callable=AsyncMock),
        pytest.raises(ValueError, match="custom graph_model is only supported"),
    ):
        await cognify_mod.cognify(graph_model=_schema(), extractor="gliner_demo")


@pytest.mark.asyncio
async def test_cognify_dry_run_estimates_with_the_built_class(local_cognify):
    with patch.object(
        estimator_mod, "estimate_cognify_dry_run", new_callable=AsyncMock
    ) as estimate:
        await cognify_mod.cognify(graph_model=_schema(), dry_run=True, extractor="llm")

    _assert_is_the_staff_graph(estimate.await_args.kwargs["graph_model"])
    local_cognify.executor.assert_not_called()


# ----- remember(): the dict travels to cognify() unchanged -----


@pytest.fixture
def local_remember():
    """remember() with add() and cognify() replaced: what they are awaited with is kept."""
    user = MagicMock(id=uuid4())
    dataset = MagicMock(id=uuid4())
    with (
        patch("cognee.shared.utils.send_telemetry"),
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch("cognee.modules.engine.operations.setup.setup", AsyncMock()) as setup,
        patch("cognee.modules.users.methods.get_default_user", AsyncMock(return_value=user)),
        patch.object(
            remember_mod,
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(user, [dataset])),
        ),
        patch("cognee.api.v1.add.add", AsyncMock()) as add,
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={})) as cognify,
    ):
        yield SimpleNamespace(add=add, cognify=cognify, setup=setup)


@pytest.mark.asyncio
@pytest.mark.parametrize("run_in_background", [False, True])
async def test_remember_hands_the_schema_dict_to_cognify_unchanged(
    local_remember, run_in_background
):
    schema = _schema()
    snapshot = json.loads(json.dumps(schema))

    result = await remember_mod.remember(
        "Ada works with Grace.",
        graph_model=schema,
        self_improvement=False,
        run_in_background=run_in_background,
    )
    if run_in_background:
        await result

    local_remember.add.assert_awaited_once()
    assert local_remember.cognify.await_args.kwargs["graph_model"] is schema
    assert schema == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("run_in_background", [False, True])
@pytest.mark.parametrize(
    "case, schema, error, text", INVALID_SCHEMAS, ids=[c[0] for c in INVALID_SCHEMAS]
)
async def test_remember_refuses_an_invalid_schema_before_storing(
    local_remember, run_in_background, case, schema, error, text
):
    with pytest.raises(error, match=text):
        await remember_mod.remember(
            "Ada works with Grace.",
            graph_model=schema(),
            run_in_background=run_in_background,
        )

    local_remember.setup.assert_not_awaited()
    local_remember.add.assert_not_awaited()
    local_remember.cognify.assert_not_awaited()


@pytest.mark.asyncio
async def test_remember_dry_run_estimates_with_the_built_class(local_remember):
    with (
        patch("cognee.infrastructure.llm.resolve_chunk_size", AsyncMock(return_value=512)),
        patch.object(
            estimator_mod, "estimate_remember_dry_run", new_callable=AsyncMock
        ) as estimate,
    ):
        await remember_mod.remember(
            "Ada works with Grace.", graph_model=_schema(), dry_run=True, extractor="llm"
        )

    _assert_is_the_staff_graph(estimate.await_args.kwargs["graph_model"])
    local_remember.add.assert_not_awaited()


# ----- the routes forward the caller's schema -----


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_cognify_router(), prefix="/cognify")
    app.include_router(get_remember_router(), prefix="/remember")
    user = SimpleNamespace(id=uuid4(), email="t@example.com", is_active=True, tenant_id=uuid4())

    async def override_user():
        return user

    app.dependency_overrides[get_authenticated_user] = override_user

    @app.exception_handler(CogneeApiError)
    async def cognee_error_handler(_, exc: CogneeApiError):
        # The body cognee/api/client.py renders for a CogneeApiError.
        return JSONResponse(
            status_code=exc.status_code, content={"detail": f"{exc.message} [{exc.name}]"}
        )

    return TestClient(app)


@pytest.fixture
def restore_api_packages():
    saved = [(pkg, getattr(pkg, name)) for pkg, name in _API_FUNCTIONS]
    yield
    for (pkg, name), (_, original) in zip(_API_FUNCTIONS, saved):
        setattr(pkg, name, original)


_API_FUNCTIONS = [(add_pkg, "add"), (cognify_pkg, "cognify"), (remember_pkg, "remember")]


def _remember_form(graph_model_text: str | None, run_in_background: bool = False):
    form = {"datasetName": "ds", "run_in_background": str(run_in_background).lower()}
    if graph_model_text is not None:
        form["graph_model"] = graph_model_text
    return form


_UPLOAD = {"data": ("a.txt", b"Ada works with Grace.", "text/plain")}


def test_cognify_route_forwards_the_schema_dict(client, restore_api_packages):
    cognify_pkg.cognify = AsyncMock(return_value={})
    schema = _schema()

    resp = client.post("/cognify", json={"datasets": ["ds"], "graph_model": schema})

    assert resp.status_code == 200
    assert cognify_pkg.cognify.await_args.kwargs["graph_model"] == schema


@pytest.mark.parametrize("graph_model", [None, {}])
def test_cognify_route_sends_no_model_when_none_was_given(
    client, restore_api_packages, graph_model
):
    """Missing and ``{}`` both mean the default: cognify() is left to apply its own."""
    cognify_pkg.cognify = AsyncMock(return_value={})
    body = {"datasets": ["ds"]} | ({} if graph_model is None else {"graph_model": graph_model})

    resp = client.post("/cognify", json=body)

    assert resp.status_code == 200
    assert "graph_model" not in cognify_pkg.cognify.await_args.kwargs


class _RememberResultStub:
    status = "completed"

    def to_dict(self):
        return {"status": "completed"}


def test_remember_route_forwards_the_schema_dict(client, restore_api_packages):
    remember_pkg.remember = AsyncMock(return_value=_RememberResultStub())
    schema = _schema()

    resp = client.post("/remember", data=_remember_form(json.dumps(schema)), files=_UPLOAD)

    assert resp.status_code == 200
    assert remember_pkg.remember.await_args.kwargs["graph_model"] == schema


def test_remember_route_sends_no_model_when_the_field_is_absent(client, restore_api_packages):
    remember_pkg.remember = AsyncMock(return_value=_RememberResultStub())

    resp = client.post("/remember", data=_remember_form(None), files=_UPLOAD)

    assert resp.status_code == 200
    assert "graph_model" not in remember_pkg.remember.await_args.kwargs


@pytest.fixture
def no_storage(restore_api_packages):
    """Real cognify() and remember(); add(), the setup and the pipeline must stay untouched."""
    add_pkg.add = AsyncMock()
    with (
        patch("cognee.modules.engine.operations.setup.setup", AsyncMock()) as setup,
        patch.object(cognify_mod, "get_pipeline_executor") as executor,
        patch(
            "cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock
        ) as migrations,
    ):
        yield SimpleNamespace(
            add=add_pkg.add, setup=setup, executor=executor, migrations=migrations
        )


def _assert_nothing_ran(no_storage):
    no_storage.add.assert_not_awaited()
    no_storage.setup.assert_not_awaited()
    no_storage.migrations.assert_not_awaited()
    no_storage.executor.assert_not_called()


@pytest.mark.parametrize("run_in_background", [False, True])
def test_remember_route_refuses_invalid_json_text(client, no_storage, run_in_background):
    resp = client.post(
        "/remember", data=_remember_form("{not json", run_in_background), files=_UPLOAD
    )

    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("graph_model is not valid JSON")
    _assert_nothing_ran(no_storage)


@pytest.mark.parametrize("run_in_background", [False, True])
@pytest.mark.parametrize(
    "case, schema, error, text", INVALID_SCHEMAS, ids=[c[0] for c in INVALID_SCHEMAS]
)
def test_remember_route_refuses_an_invalid_schema(
    client, no_storage, run_in_background, case, schema, error, text
):
    resp = client.post(
        "/remember",
        data=_remember_form(json.dumps(schema()), run_in_background),
        files=_UPLOAD,
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert text in detail
    assert detail.endswith(f"[{error.__name__}]")
    _assert_nothing_ran(no_storage)


# The cognify route reads {} as the default model, and its DTO answers a non-object
# with a 422 before the handler runs: both keep their status from before.
_COGNIFY_ROUTE_400S = [c for c in INVALID_SCHEMAS if c[0] not in ("not_an_object", "empty_object")]


@pytest.mark.parametrize("run_in_background", [False, True])
@pytest.mark.parametrize(
    "case, schema, error, text", _COGNIFY_ROUTE_400S, ids=[c[0] for c in _COGNIFY_ROUTE_400S]
)
def test_cognify_route_refuses_an_invalid_schema(
    client, no_storage, run_in_background, case, schema, error, text
):
    resp = client.post(
        "/cognify",
        json={"datasets": ["ds"], "graph_model": schema(), "run_in_background": run_in_background},
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert text in detail
    assert detail.endswith(f"[{error.__name__}]")
    _assert_nothing_ran(no_storage)


def test_cognify_route_answers_a_non_object_schema_with_its_dto_422(client, no_storage):
    resp = client.post("/cognify", json={"datasets": ["ds"], "graph_model": [1, 2]})

    assert resp.status_code == 422
    _assert_nothing_ran(no_storage)

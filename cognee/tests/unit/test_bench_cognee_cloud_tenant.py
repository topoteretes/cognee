"""Tenant creation in the cloud perf harness must survive a lost response.

POST /api/v1/tenants commits the tenant row before provisioning starts, so
when the response is lost (intermediary reset at ~272s, or aiohttp's 300s
timeout) a retry re-POSTs a name that now exists and gets a 503. That was
4 of the last 5 nightly cloud-arm failures. These tests drive the real
_create_cloud_tenant against an in-process aiohttp controller so a lost
response is a genuine transport close, not a mocked exception.
"""

import asyncio
import importlib.util
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

_BENCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "performance"
    / "statistics_percentile"
    / "bench_cognee.py"
)
_SPEC = importlib.util.spec_from_file_location("bench_cognee", _BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

T1 = "aaaaaaaa-0000-0000-0000-000000000001"
T2 = "bbbbbbbb-0000-0000-0000-000000000002"


class FakeController:
    """Enough of the tenant-controller API for _create_cloud_tenant.

    `tenants` is name -> (id, status). `drop_first_post_response` commits the
    row and then closes the transport without answering, which is exactly
    what the nightly saw. `rollback_on_poll` deletes T1 on the N-th status
    poll and re-creates the same name as T2 — the controller's own tenacity
    retry after a provisioning rollback.
    """

    def __init__(self):
        self.tenants: dict[str, tuple[str, str]] = {}
        self.post_calls: list[str] = []
        self.status_polls = 0
        self.drop_first_post_response = False
        self.always_503 = False
        self.reject_400 = False
        self.rollback_on_poll: int | None = None

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/api/v1/tenants", self.create)
        app.router.add_get("/api/v1/tenants/me/with-status", self.list_mine)
        app.router.add_get("/api/v1/tenants/{tid}/status", self.status)
        return app

    async def create(self, request: web.Request):
        name = request.query["tenant_name"]
        self.post_calls.append(name)
        if self.reject_400:
            return web.json_response({"detail": "Invalid tenant name"}, status=400)
        if name in self.tenants or self.always_503:
            return web.json_response(
                {
                    "detail": f"Tenant with name '{name}' already exists for the user. "
                    "[TenantCreationError]"
                },
                status=503,
            )
        tid = T1 if not self.tenants else str(uuid.uuid4())
        self.tenants[name] = (tid, "healthy")  # committed BEFORE the response
        if self.drop_first_post_response:
            self.drop_first_post_response = False
            request.transport.close()  # the response is lost on the wire
            await asyncio.sleep(0)
            raise web.HTTPInternalServerError()  # never reaches the client
        return web.json_response({"tenant_id": tid})

    async def list_mine(self, request: web.Request):
        return web.json_response(
            [{"id": tid, "name": name, "status": st} for name, (tid, st) in self.tenants.items()]
        )

    async def status(self, request: web.Request):
        self.status_polls += 1
        tid = request.match_info["tid"]
        if self.rollback_on_poll is not None and self.status_polls == self.rollback_on_poll:
            # rollback deletes T1, then the controller's @retry re-creates the
            # SAME name under a NEW id
            for name, (existing, _) in list(self.tenants.items()):
                if existing == tid:
                    self.tenants[name] = (T2, "healthy")
        for existing, st in self.tenants.values():
            if existing == tid:
                return web.json_response({"status": st})
        return web.json_response({"detail": "not found"}, status=404)


@pytest_asyncio.fixture
async def controller():
    ctrl = FakeController()
    server = TestServer(ctrl.app())
    await server.start_server()
    try:
        yield ctrl, f"http://{server.host}:{server.port}"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_lost_response_must_not_double_post(controller):
    """The row was committed; a second POST gets the 503. Adopt instead."""
    ctrl, url = controller
    ctrl.drop_first_post_response = True
    tid, _, _ = await bench._create_cloud_tenant(url, "k", "bench-x-1", ready_timeout_s=5)
    assert tid == T1
    assert ctrl.post_calls == ["bench-x-1"], "a lost response must not be retried as a re-POST"


@pytest.mark.asyncio
async def test_already_exists_is_adopted(controller):
    ctrl, url = controller
    ctrl.tenants["bench-x-2"] = (T1, "healthy")
    tid, _, _ = await bench._create_cloud_tenant(url, "k", "bench-x-2", ready_timeout_s=5)
    assert tid == T1


@pytest.mark.asyncio
async def test_readiness_timeout_reports_the_tenant_for_cleanup(controller):
    """The leak test: a tenant that never goes healthy must still be torn down."""
    ctrl, url = controller
    ctrl.tenants["bench-x-3"] = (T1, "provisioning")
    with pytest.raises(bench.TenantCreateFailed) as exc:
        await bench._create_cloud_tenant(url, "k", "bench-x-3", ready_timeout_s=0.2)
    assert exc.value.tenant_id == T1, f"no tenant_id on {exc.value!r} -> leak"


@pytest.mark.asyncio
async def test_controller_rollback_and_retry_is_followed_not_leaked(controller):
    """404 on the id we hold does not mean nothing exists: follow the re-create."""
    ctrl, url = controller
    ctrl.tenants["bench-x-4"] = (T1, "provisioning")
    ctrl.rollback_on_poll = 2
    tid, _, _ = await bench._create_cloud_tenant(url, "k", "bench-x-4", ready_timeout_s=10)
    assert tid == T2


@pytest.mark.asyncio
async def test_genuine_rejection_fails_immediately(controller):
    """A 400 is a real rejection: no retry, no adoption, nothing to clean up."""
    ctrl, url = controller
    ctrl.reject_400 = True
    with pytest.raises(bench.TenantCreateFailed) as exc:
        await bench._create_cloud_tenant(url, "k", "bench-x-5", ready_timeout_s=5)
    assert "(400)" in str(exc.value)
    assert exc.value.tenant_id is None
    assert ctrl.post_calls == ["bench-x-5"]


@pytest.mark.asyncio
async def test_create_failure_still_tears_the_tenant_down(monkeypatch):
    """run_benchmark_cloud's teardown is gated on tenant_id; the failing branch must set it."""
    deleted = []

    async def _boom(*a, **k):
        raise bench.TenantCreateFailed("boom", tenant_id="T1")

    async def _delete(management_url, api_key, tenant_id):
        deleted.append(tenant_id)
        return 0.01

    monkeypatch.setattr(bench, "_create_cloud_tenant", _boom)
    monkeypatch.setattr(bench, "_delete_cloud_tenant", _delete)
    result = await bench.run_benchmark_cloud(
        [{"title": "t", "content": "c"}],
        config={
            "create_tenant": True,
            "tenant_url": "https://unused",
            "management_url": "https://unused",
            "tenant_api_key": "k",
        },
    )
    assert deleted == ["T1"]
    assert result["success"] is False

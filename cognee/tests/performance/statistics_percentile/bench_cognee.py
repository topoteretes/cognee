"""
Standalone Cognee ingest benchmark.

Inserts text memories into Cognee (add + cognify) and measures wall-clock
time for each phase.

Usage:
    python bench_cognee.py                     # default settings
    python bench_cognee.py --memories data.json # custom memories file
    python bench_cognee.py --llm-model gpt-4o  # override LLM model
    python bench_cognee.py --tenant-url https://tenant-x.aws.cognee.ai --tenant-api-key ck_...
                                               # run against a Cognee Cloud tenant

The memories file should be a JSON array of objects with "title" and "content"
keys (see the bundled memories.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

# Mock loading/replay lives in the shared mock-ingestion module
# (cognee/tests/utils/mock_ingestion) so other suites — e.g. the large-scale
# migration release test — build systems the exact same way. The private
# aliases keep this file's call sites and capture_mock.py's imports stable.
from cognee.tests.utils.mock_ingestion import (
    install_mocks as _install_mocks,
    load_memories,
    load_mock_data as _load_mock_data,
    memory_to_text,
)

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MEMORIES_FILE = Path(__file__).with_name("memories.json")
DEFAULT_MOCK_MEMORIES_FILE = Path(__file__).with_name("mock_memories.json")
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMS = 1536
DATASET_NAME = "bench_memories"
SEARCH_QUERY = "What is in the document"

# Search types timed in Phase 3, in run order: `(metric suffix, SearchType name)`.
# Each gets its own metric so the nightly report can compare retrieval
# strategies side by side — GRAPH_COMPLETION is the default graph traversal,
# HYBRID_COMPLETION the chunk+entity blend. Both the local and the cloud
# benchmark iterate this list, so the two modes always report the same types.
BENCHMARKED_SEARCH_TYPES = (
    ("graph_completion", "GRAPH_COMPLETION"),
    ("hybrid_completion", "HYBRID_COMPLETION"),
)

ENV_FILE = Path(__file__).resolve().parents[4] / ".env"

os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")


def _resolve_config(args: argparse.Namespace) -> dict:
    """Resolve config values: CLI arg → .env file → script defaults."""
    mock_llm = getattr(args, "mock_llm", False)
    env = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}

    def pick(cli_val, env_key: str, default):
        if cli_val is not None:
            return cli_val
        env_val = env.get(env_key) or os.environ.get(env_key)
        if env_val is not None:
            return type(default)(env_val) if not isinstance(default, str) else env_val
        return default

    api_key = pick(None, "LLM_API_KEY", "") or pick(None, "OPENAI_API_KEY", "")
    if not api_key and not mock_llm:
        sys.exit("Error: LLM_API_KEY is not set (CLI, .env, or environment)")

    # Embeddings may use a different provider/key than the LLM (e.g.
    # + OpenAI embeddings). Resolve the embedding key independently, falling back
    # to the LLM/OpenAI key when LLM and embeddings share a provider.
    embedding_api_key = pick(None, "EMBEDDING_API_KEY", "") or api_key

    return {
        "api_key": api_key or "mock-key",
        "embedding_api_key": embedding_api_key or "mock-key",
        "llm_provider": pick(args.llm_provider, "LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
        "llm_model": pick(args.llm_model, "LLM_MODEL", DEFAULT_LLM_MODEL),
        "embedding_provider": pick(
            args.embedding_provider, "EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER
        ),
        "embedding_model": pick(args.embedding_model, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "embedding_dims": pick(args.embedding_dims, "EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMS),
        "mock_llm": mock_llm,
    }


def _resolve_cloud_config(args: argparse.Namespace) -> dict:
    """Resolve config for cloud mode (--tenant-url / --create-tenant):
    all processing is server-side."""
    if args.mock_llm:
        sys.exit("Error: --mock-llm is not supported in cloud mode (LLM runs server-side)")
    api_key = args.tenant_api_key or os.environ.get("COGNEE_API_KEY", "")
    if not api_key:
        sys.exit("Error: --tenant-api-key (or COGNEE_API_KEY env var) is required in cloud mode")
    if not args.tenant_url and not args.create_tenant:
        sys.exit("Error: cloud mode needs --tenant-url or --create-tenant")
    return {
        "tenant_url": args.tenant_url.rstrip("/") if args.tenant_url else None,
        "tenant_api_key": api_key,
        "dataset_name": args.dataset_name or DATASET_NAME,
        "create_tenant": bool(args.create_tenant),
        "management_url": args.management_url.rstrip("/"),
        "mock_llm": False,
    }


# ── Mock LLM / Embedding ────────────────────────────────────────────────────


# Stats for --mock-document-embeddings, surfaced in the results JSON:
# how many embed inputs were served from the store vs embedded live.
_DOC_EMBED_STATS = {"served": 0, "embedded_live": 0}


def _install_document_embedding_mock(embeddings_file: Path) -> None:
    """Replay captured document embeddings; embed anything unseen for REAL.

    The `mock_document_embeddings` system: every text captured during the
    real cognify run (chunks, entities, edge texts, summaries) gets its
    stored REAL vector back with no API call, while unseen texts — search
    queries — go through the live embedding API. Vector search therefore
    behaves exactly like production, without re-embedding the corpus.

    Requires a working embedding API key at replay time for the live path.
    """
    import importlib

    with open(embeddings_file) as f:
        payload = json.load(f)
    stored_vectors: dict[str, list[float]] = payload["vectors"]
    print(
        f"Document-embedding mock enabled: {len(stored_vectors)} stored vectors "
        f"(model: {payload.get('model')}); unseen texts embed live"
    )

    from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
        get_embedding_engine,
    )

    engine_cls = type(get_embedding_engine())
    original = engine_cls.embed_text

    async def replay_embed(self, text):
        missing = [item for item in text if item not in stored_vectors]
        if missing:
            fetched = await original(self, missing)
            for item, vector in zip(missing, fetched):
                stored_vectors[item] = vector  # in-memory only; repeat queries hit once
            _DOC_EMBED_STATS["embedded_live"] += len(missing)
        _DOC_EMBED_STATS["served"] += len(text) - len(missing)
        return [stored_vectors[item] for item in text]

    engine_cls.embed_text = replay_embed

    # Clear cached engines so any instance constructed before this install is
    # replaced by one using the patched class (class-level patch covers both,
    # but the caches may also hold pre-mock env-derived config).
    emb_mod = importlib.import_module(
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine"
    )
    vec_mod = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    emb_mod.create_embedding_engine.cache_clear()
    vec_mod._create_vector_engine.cache_clear()


# ── Helpers ──────────────────────────────────────────────────────────────────


# ── Benchmark ────────────────────────────────────────────────────────────────


async def run_benchmark(
    memories: list[dict],
    *,
    config: dict,
) -> dict:
    import cognee

    # Register community adapters before any engine is created. Comma-separated
    # module names; a module-level register() is called if present (some
    # adapters register on import alone).
    import importlib

    for module_name in filter(None, os.environ.get("COGNEE_REGISTER_ADAPTERS", "").split(",")):
        module = importlib.import_module(module_name)
        register = getattr(module, "register", None)
        if callable(register):
            register()
        print(f"Registered adapter module: {module_name}")

    llm_model = config["llm_model"]
    llm_provider = config["llm_provider"]
    embedding_model = config["embedding_model"]
    embedding_dims = config["embedding_dims"]

    cognee.config.set_llm_api_key(config["api_key"])
    cognee.config.set_llm_provider(llm_provider)
    cognee.config.set_llm_model(llm_model)
    cognee.config.set_embedding_provider(config["embedding_provider"])
    cognee.config.set_embedding_model(embedding_model)
    cognee.config.set_embedding_dimensions(embedding_dims)
    cognee.config.set_embedding_api_key(config["embedding_api_key"])

    document_embeddings_file = config.get("mock_document_embeddings_file")
    if config.get("mock_llm"):
        mock_data = _load_mock_data(config["mock_memories_file"])
        # With document-embedding replay active, embeddings are handled there;
        # only the LLM gets the built-in mock treatment.
        _install_mocks(mock_data, mock_embeddings=not document_embeddings_file)
        print(
            "Mock LLM mode enabled"
            if document_embeddings_file
            else "Mock LLM/embedding mode enabled"
        )
    if document_embeddings_file:
        _install_document_embedding_mock(Path(document_embeddings_file))

    n = len(memories)
    status = {
        "prune": "success",
        "db_setup": "success",
        "add": "success",
        "cognify": "success",
        "dataset_delete": "success",
        **{f"search_{key}": "success" for key, _ in BENCHMARKED_SEARCH_TYPES},
    }
    t_prune = 0.0
    t_db_setup = 0.0
    t_add = 0.0
    t_cognify = 0.0
    t_search = {key: 0.0 for key, _ in BENCHMARKED_SEARCH_TYPES}
    t_dataset_delete = 0.0

    # ── Prune (clean slate) ──────────────────────────────────────────────
    print("Pruning previous data...")
    try:
        t_prune_start = time.time()
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        t_prune = time.time() - t_prune_start
        print(f"  Prune completed in {t_prune:.2f}s")
    except Exception as e:
        t_prune = time.time() - t_prune_start
        status["prune"] = f"failed: {e}"
        print(f"  Prune FAILED: {e}")

    # ── DB Setup ─────────────────────────────────────────────────────────
    try:
        from cognee.modules.engine.operations.setup import setup

        t_db_setup_start = time.time()
        await setup()
        t_db_setup = time.time() - t_db_setup_start
    except Exception as e:
        t_db_setup = time.time() - t_db_setup_start
        status["db_setup"] = f"failed: {e}"
        print(f"  DB setup FAILED: {e}")

    # ── Phase 1: cognee.add() ────────────────────────────────────────────
    print(f"\nPhase 1: Adding {n} memories via cognee.add()...")
    text_list = [memory_to_text(mem) for mem in memories]

    try:
        t_add_start = time.time()
        await cognee.add(text_list, dataset_name=DATASET_NAME)
        t_add = time.time() - t_add_start
    except Exception as e:
        t_add = time.time() - t_add_start
        status["add"] = f"failed: {e}"
        print(f"  Add FAILED: {e}")

    # ── Phase 2: cognee.cognify() ────────────────────────────────────────
    print("\nPhase 2: Running cognee.cognify() (knowledge graph build)...")
    try:
        t_cognify_start = time.time()
        await cognee.cognify(data_per_batch=n, chunks_per_batch=10000)
        t_cognify = time.time() - t_cognify_start
    except Exception as e:
        t_cognify = time.time() - t_cognify_start
        status["cognify"] = f"failed: {e}"
        print(f"  Cognify FAILED: {e}")

    t_total = t_add + t_cognify

    # ── Phase 3: cognee.search() ─────────────────────────────────────────
    # One timing per benchmarked search type. `only_context=True` keeps every
    # number a measure of retrieval rather than answer generation, so the types
    # stay comparable to each other and to the previous single-search metric.
    if not _ingest_succeeded(status):
        _skip_search_phase(status, t_search)
    else:
        print("\nPhase 3: Running search queries...")
        for metric_key, search_type in BENCHMARKED_SEARCH_TYPES:
            t_q_start = time.time()
            try:
                await cognee.search(
                    query_text=SEARCH_QUERY,
                    query_type=cognee.SearchType[search_type],
                    only_context=True,
                )
                t_search[metric_key] = time.time() - t_q_start
            except Exception as e:
                t_search[metric_key] = time.time() - t_q_start
                status[f"search_{metric_key}"] = f"failed: {_err(e)}"
                print(f"  Search {search_type} FAILED: {_err(e)}")

    # ── Phase 4: dataset delete (populated) ──────────────────────────────
    # Deleting the dataset AFTER the graph is built measures the meaningful
    # case (nodes, edges, and vectors all present) — mirrors the cloud mode's
    # dataset_delete_time_s metric.
    print("\nPhase 4: Deleting the populated dataset...")
    t_dataset_delete_start = time.time()
    try:
        from cognee.api.v1.datasets.datasets import datasets as datasets_api
        from cognee.modules.data.methods import get_datasets_by_name
        from cognee.modules.users.methods import get_default_user

        user = await get_default_user()
        found = await get_datasets_by_name([DATASET_NAME], user.id)
        if found:
            await datasets_api.empty_dataset(found[0].id, user)
        t_dataset_delete = time.time() - t_dataset_delete_start
        print(f"  Dataset deleted in {t_dataset_delete:.2f}s")
    except Exception as e:
        t_dataset_delete = time.time() - t_dataset_delete_start
        status["dataset_delete"] = f"failed: {e}"
        print(f"  Dataset delete FAILED: {e}")

    # "skipped" only ever appears when add/cognify already carries "failed:",
    # so this changes no pass/fail outcome -- it only stops a skipped search
    # from being counted as a second failure.
    all_ok = all(v in ("success", "skipped") for v in status.values())

    # ── Report ───────────────────────────────────────────────────────────
    results = {
        "memories_count": n,
        "add_time_s": round(t_add, 3),
        "cognify_time_s": round(t_cognify, 3),
        "total_ingest_time_s": round(t_total, 3),
        "prune_time_s": round(t_prune, 3),
        "db_setup_time_s": round(t_db_setup, 3),
        # A skipped search has no timing. Omit it rather than write 0.0 --
        # the file's own rule is elapsed-until-failure, never a fabricated
        # zero, and the report tolerates a missing metric.
        **{
            f"search_time_{key}": t_search[key]
            for key, _ in BENCHMARKED_SEARCH_TYPES
            if t_search.get(key) is not None
        },
        "dataset_delete_time_s": round(t_dataset_delete, 3),
        "status": status,
        "success": all_ok,
        "config": {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dims,
            "dataset_name": DATASET_NAME,
            "mock_llm": config.get("mock_llm", False),
            "mock_document_embeddings": bool(document_embeddings_file),
        },
    }
    if document_embeddings_file:
        results["document_embedding_stats"] = dict(_DOC_EMBED_STATS)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Memories inserted : {n}")
    print(f"  cognee.add() time : {t_add:.2f}s  ({t_add / n:.2f}s per memory)  [{status['add']}]")
    print(f"  cognify() time    : {t_cognify:.2f}s  [{status['cognify']}]")
    print(f"  Total ingest time : {t_total:.2f}s  ({t_total / n:.2f}s per memory)")
    for metric_key, search_type in BENCHMARKED_SEARCH_TYPES:
        timing = t_search.get(metric_key)
        shown = f"{timing:.2f}s" if timing is not None else "n/a  "
        print(f"  Search {search_type:<18}: {shown}  [{status[f'search_{metric_key}']}]")
    print(f"  DB setup time     : {t_db_setup:.2f}s  [{status['db_setup']}]")
    print(f"  Prune time        : {t_prune:.2f}s  [{status['prune']}]")
    print(f"  Dataset delete    : {t_dataset_delete:.2f}s  [{status['dataset_delete']}]")
    print(f"  LLM model         : {llm_model}")
    print(f"  Embedding model   : {embedding_model} ({embedding_dims}d)")
    if config.get("mock_llm"):
        print("  Mock mode         : ON")
    if document_embeddings_file:
        print(
            f"  Doc-embedding mock: served {_DOC_EMBED_STATS['served']} from store, "
            f"embedded {_DOC_EMBED_STATS['embedded_live']} live"
        )
    print(f"  Overall           : {'ALL OK' if all_ok else 'SOME FAILURES'}")
    print("=" * 60)

    return results


def _err(e: Exception) -> str:
    """Readable error text: timeout-class exceptions stringify to '' (the
    infamous blank 'cognify: failed: ' in CI logs), so fall back to repr."""
    return str(e) or repr(e)


def _ingest_succeeded(status: dict) -> bool:
    """Phase 3 only means something on a graph that was actually built."""
    return status.get("add") == "success" and status.get("cognify") == "success"


def _skip_search_phase(status: dict, t_search: dict) -> None:
    """Record every benchmarked search as skipped, with no timing.

    A search against a graph that never got built returns fast without
    raising, so without this the phase reports SUCCESS and a real-looking
    float for work it did not do -- and those floats became the p50/p90/p99
    uploaded to S3 and posted to Slack (job 100109026857: three 402s on
    cognify, each followed by 'Search GRAPH_COMPLETION : 2.30s [success]').
    """
    print("\nPhase 3: SKIPPED (ingest did not succeed)")
    for metric_key, _ in BENCHMARKED_SEARCH_TYPES:
        status[f"search_{metric_key}"] = "skipped"
        t_search[metric_key] = None


# Transient-reset retries for tenant creation (see _create_cloud_tenant).
_TENANT_CREATE_ATTEMPTS = 3

# The POST can die with its tenant already committed server-side (the row and
# its UserTenant membership are committed BEFORE provisioning starts), so a
# retry re-POSTs a name that now exists and gets a 503. This timeout does NOT
# fix that -- it only shortens the wasted wait, because half the observed deaths
# are an intermediary reset at ~272s, below aiohttp's 300s default.
_TENANT_POST_TIMEOUT_S = 120.0


class TenantCreateFailed(RuntimeError):
    """Tenant creation failed. Carries the id of anything that exists so the
    caller's teardown can still reach it."""

    def __init__(self, message: str, tenant_id: str | None = None):
        super().__init__(message)
        self.tenant_id = tenant_id


async def _find_tenant_by_name(session, management_url: str, name: str) -> str | None:
    """Resolve a tenant id by name.

    GET /api/v1/tenants/me/with-status issues the byte-identical query to the
    duplicate check that produces the 503, so whenever that 503 fires this
    lookup must find the tenant.
    """
    async with session.get(f"{management_url}/api/v1/tenants/me/with-status") as resp:
        if resp.status >= 400:
            return None
        matches = [t["id"] for t in (await resp.json()) if t.get("name") == name]
    if len(matches) > 1:
        print(
            f"  WARNING: {len(matches)} tenants named '{name}' -- the controller's "
            f"check-then-act is not atomic; taking {matches[0]}",
            flush=True,
        )
    return matches[0] if matches else None


async def _create_cloud_tenant(
    management_url: str, api_key: str, tenant_name: str, ready_timeout_s: float = 600.0
) -> tuple[str, str, float]:
    """Create a tenant via the tenant-controller API and wait until healthy.

    Returns (tenant_id, service_url, seconds) where seconds covers the create
    call PLUS the wait until the tenant reports healthy — the number that
    matters is "time until a usable tenant", not just the POST round trip.
    """
    import aiohttp
    from urllib.parse import urlsplit

    t0 = time.time()
    async with aiohttp.ClientSession(headers={"X-Api-Key": api_key}) as session:
        # The POST commits the tenant row BEFORE provisioning starts, so a lost
        # response leaves a real tenant behind. Both failure shapes therefore
        # resolve by name and adopt rather than re-POSTing:
        #   * transport death (reset at ~272-294s, or aiohttp's 300s timeout)
        #   * HTTP 503 "already exists" from our own previous attempt
        # A genuine rejection (400 bad name, 401/403) still fails immediately.
        last_exc = None
        tenant_id = None
        for attempt in range(_TENANT_CREATE_ATTEMPTS):
            try:
                async with session.post(
                    f"{management_url}/api/v1/tenants",
                    params={"tenant_name": tenant_name},
                    timeout=aiohttp.ClientTimeout(total=_TENANT_POST_TIMEOUT_S),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        if resp.status == 409 or "already exists" in body:
                            existing = await _find_tenant_by_name(
                                session, management_url, tenant_name
                            )
                            if existing is not None:
                                print(
                                    f"  Tenant '{tenant_name}' already exists -- adopting {existing}",
                                    flush=True,
                                )
                                tenant_id = existing
                                break
                        raise TenantCreateFailed(f"Tenant creation failed ({resp.status}): {body}")
                    tenant_id = (await resp.json())["tenant_id"]
                break
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
                last_exc = exc
                existing = await _find_tenant_by_name(session, management_url, tenant_name)
                if existing is not None:
                    print(
                        f"  Tenant create lost its response ({_err(exc)}); adopting {existing}",
                        flush=True,
                    )
                    tenant_id = existing
                    break
                if attempt == _TENANT_CREATE_ATTEMPTS - 1:
                    raise TenantCreateFailed(f"Tenant creation failed: {_err(exc)}") from exc
                backoff = 2**attempt
                print(
                    f"  Tenant creation attempt {attempt + 1}/{_TENANT_CREATE_ATTEMPTS} failed "
                    f"({_err(exc)}); retrying in {backoff}s",
                    flush=True,
                )
                await asyncio.sleep(backoff)
        if tenant_id is None:  # pragma: no cover - defensive
            raise TenantCreateFailed(f"Tenant creation failed: {_err(last_exc)}")

        deadline = time.time() + ready_timeout_s
        while True:
            # A transient reset while polling is not "unhealthy" -- keep polling
            # until the deadline rather than failing the run on one bad packet.
            try:
                async with session.get(
                    f"{management_url}/api/v1/tenants/{tenant_id}/status"
                ) as resp:
                    if resp.status in (403, 404):
                        # The id we hold is gone. That does NOT mean nothing
                        # exists: the controller's rollback deletes the row and
                        # its own @retry then re-creates the SAME NAME under a
                        # NEW id, which may go healthy. Re-resolve before giving
                        # up, or we abandon a live, billed tenant.
                        replacement = await _find_tenant_by_name(
                            session, management_url, tenant_name
                        )
                        if replacement is not None and replacement != tenant_id:
                            print(
                                f"  Tenant {tenant_id} was rolled back; controller re-created "
                                f"'{tenant_name}' as {replacement} -- following it",
                                flush=True,
                            )
                            tenant_id = replacement
                        elif replacement is None and time.time() > deadline:
                            raise TenantCreateFailed(
                                f"Tenant {tenant_id} disappeared while waiting "
                                f"(status {resp.status})",
                                tenant_id=None,
                            )
                    elif resp.status < 400:
                        reported = (await resp.json()).get("status")
                        if reported == "healthy":
                            break
                        if reported == "failed":
                            raise TenantCreateFailed(
                                f"Tenant {tenant_id} provisioning failed", tenant_id=tenant_id
                            )
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
                if time.time() > deadline:
                    raise TenantCreateFailed(
                        f"Tenant {tenant_id} not healthy after {ready_timeout_s:.0f}s "
                        f"(last error: {_err(exc)})",
                        tenant_id=await _find_tenant_by_name(session, management_url, tenant_name),
                    ) from exc
            if time.time() > deadline:
                # A fresh by-name lookup rather than the possibly-stale local
                # id, so teardown deletes whatever actually exists.
                raise TenantCreateFailed(
                    f"Tenant {tenant_id} not healthy after {ready_timeout_s:.0f}s",
                    tenant_id=await _find_tenant_by_name(session, management_url, tenant_name),
                )
            await asyncio.sleep(2)

    # Service URL convention: api.<domain> hosts the controller and
    # tenant-<id>.<domain> hosts the tenant (e.g. https://tenant-<id>.aws.cognee.ai).
    management_host = urlsplit(management_url).netloc
    service_host = management_host.replace("api.", f"tenant-{tenant_id}.", 1)
    return tenant_id, f"https://{service_host}", time.time() - t0


async def _delete_cloud_tenant(management_url: str, api_key: str, tenant_id: str) -> float:
    """Delete a benchmark-created tenant; returns seconds taken."""
    import aiohttp

    t0 = time.time()
    async with aiohttp.ClientSession(headers={"X-Api-Key": api_key}) as session:
        async with session.delete(
            f"{management_url}/api/v1/tenants", params={"tenant_id": tenant_id}
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Tenant deletion failed ({resp.status}): {body}")
    return time.time() - t0


async def _delete_cloud_dataset_if_exists(client, dataset_name: str) -> None:
    """Dataset-scoped clean slate for cloud runs.

    Multiple benchmark suites share one tenant in nightly CI, so a global
    ``forget(everything=True)`` would wipe a concurrently-running suite's
    data. Delete ONLY this suite's dataset — and only when it exists (the
    first run on a fresh tenant has nothing to delete).
    """
    session = await client._get_session()
    async with session.get(f"{client.service_url}/api/v1/datasets") as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(f"Listing datasets failed ({resp.status}): {body}")
        datasets = await resp.json()
    existing_names = {record.get("name") for record in datasets if isinstance(record, dict)}
    if dataset_name not in existing_names:
        print(f"  Dataset '{dataset_name}' not present on tenant; nothing to delete")
        return
    await client.forget(dataset=dataset_name)
    print(f"  Deleted dataset '{dataset_name}'")


async def _wait_for_cloud_cognify(
    client, cognify_response: dict, poll_interval_s: float = 5.0, timeout_s: float = 7200.0
) -> None:
    """Block until every dataset in a cognify response finishes building.

    Cloud tenants run cognify in the background and return
    ``PipelineRunStarted`` immediately, so the POST duration alone measures
    only the API round trip. Poll ``GET /v1/datasets/status`` until each
    started dataset reports COMPLETED (raising on ERRORED/timeout) so the
    benchmarked cognify time covers the actual knowledge-graph build.
    """
    pending = {
        str(dataset_id)
        for dataset_id, info in (cognify_response or {}).items()
        if isinstance(info, dict) and info.get("status") == "PipelineRunStarted"
    }
    if not pending:
        return  # the tenant processed synchronously; nothing to wait for

    session = await client._get_session()
    deadline = time.time() + timeout_s
    while pending:
        if time.time() > deadline:
            raise TimeoutError(f"cognify still running after {timeout_s:.0f}s: {sorted(pending)}")
        await asyncio.sleep(poll_interval_s)
        params = [("dataset", dataset_id) for dataset_id in sorted(pending)]
        params.append(("pipeline", "cognify_pipeline"))
        async with session.get(
            f"{client.service_url}/api/v1/datasets/status", params=params
        ) as resp:
            if resp.status >= 400:
                continue  # transient status-endpoint hiccup; keep polling
            statuses = await resp.json()
        for dataset_id in list(pending):
            value = statuses.get(dataset_id)
            if isinstance(value, dict):  # nested {pipeline_name: status} shape
                value = value.get("cognify_pipeline")
            if value == "DATASET_PROCESSING_COMPLETED":
                pending.discard(dataset_id)
            elif value == "DATASET_PROCESSING_ERRORED":
                raise RuntimeError(f"cognify errored on the tenant for dataset {dataset_id}")


async def run_benchmark_cloud(
    memories: list[dict],
    *,
    config: dict,
) -> dict:
    """Run the same phases against a remote Cognee tenant via cognee.serve().

    Timings include network latency; LLM/embedding config lives server-side.
    """
    import cognee

    n = len(memories)
    status = {
        "prune": "success",
        "db_setup": "success",  # server-side, nothing to set up from the client
        "add": "success",
        "cognify": "success",
        **{f"search_{key}": "success" for key, _ in BENCHMARKED_SEARCH_TYPES},
    }
    t_prune = 0.0
    t_add = 0.0
    t_cognify = 0.0
    t_search = {key: 0.0 for key, _ in BENCHMARKED_SEARCH_TYPES}
    t_tenant_create = 0.0
    t_tenant_delete = 0.0
    t_dataset_delete = 0.0

    dataset_name = config.get("dataset_name", DATASET_NAME)

    # ── Phase 0 (cloud-only metric): tenant creation ─────────────────────
    # With --create-tenant, the whole benchmark runs on a tenant provisioned
    # here, and tenant creation time (create call + wait-until-healthy) is
    # measured as its own metric. The tenant is deleted after the run.
    tenant_id = None
    tenant_url = config["tenant_url"]
    tenant_ready = True
    if config.get("create_tenant"):
        status["tenant_create"] = "success"
        status["tenant_delete"] = "success"
        # DNS-safe, unique per run: labels use underscores, hostnames cannot.
        tenant_name = f"bench-{dataset_name}-{int(time.time())}".replace("_", "-")
        print(f"Phase 0: Creating tenant '{tenant_name}'...")
        t_tenant_create_start = time.time()
        try:
            tenant_id, tenant_url, t_tenant_create = await _create_cloud_tenant(
                config["management_url"], config["tenant_api_key"], tenant_name
            )
            print(f"  Tenant {tenant_id} ready in {t_tenant_create:.2f}s at {tenant_url}")
        except Exception as e:
            # Record elapsed-until-failure like every other phase (0.0 would
            # skew failed-run percentiles low).
            t_tenant_create = time.time() - t_tenant_create_start
            # TenantCreateFailed carries the id of whatever exists; without
            # this, teardown (gated on tenant_id) was unreachable on exactly
            # the branch that leaks a real, billed tenant.
            tenant_id = getattr(e, "tenant_id", None)
            status["tenant_create"] = f"failed: {_err(e)}"
            skipped_phases = ("prune", "add", "cognify") + tuple(
                f"search_{key}" for key, _ in BENCHMARKED_SEARCH_TYPES
            )
            for phase in skipped_phases:
                status[phase] = "skipped"
            tenant_ready = False
            print(f"  Tenant creation FAILED: {_err(e)}")

    if tenant_ready:
        client = await cognee.serve(url=tenant_url, api_key=config["tenant_api_key"])

        # ── Prune (dataset-scoped clean slate on the tenant) ─────────────
        # Suites share the tenant, so never forget(everything=True) here.
        # On a just-created tenant the dataset cannot exist, so the pre-run
        # prune is skipped; dataset deletion is instead measured AFTER the
        # graph is built (dataset_delete_time_s), which is the meaningful
        # number — deleting a populated dataset.
        if config.get("create_tenant"):
            status["prune"] = "skipped"
            print("Skipping pre-run prune (fresh tenant, nothing to delete)")
        else:
            print(f"Deleting previous '{dataset_name}' dataset on tenant (if it exists)...")
            t_prune_start = time.time()
            try:
                await _delete_cloud_dataset_if_exists(client, dataset_name)
                t_prune = time.time() - t_prune_start
                print(f"  Prune completed in {t_prune:.2f}s")
            except Exception as e:
                t_prune = time.time() - t_prune_start
                status["prune"] = f"failed: {_err(e)}"
                print(f"  Prune FAILED: {_err(e)}")

        # ── Phase 1: add ─────────────────────────────────────────────────
        print(f"\nPhase 1: Adding {n} memories via remote add...")
        text_list = [memory_to_text(mem) for mem in memories]
        t_add_start = time.time()
        try:
            await client.add(text_list, dataset_name=dataset_name)
            t_add = time.time() - t_add_start
        except Exception as e:
            t_add = time.time() - t_add_start
            status["add"] = f"failed: {_err(e)}"
            print(f"  Add FAILED: {_err(e)}")

        # ── Phase 2: cognify ─────────────────────────────────────────────
        print("\nPhase 2: Running remote cognify (knowledge graph build)...")
        t_cognify_start = time.time()
        try:
            cognify_response = await client.cognify(datasets=[dataset_name])
            # The tenant builds the graph in the background — wait for
            # completion so t_cognify measures the build, not the round trip.
            await _wait_for_cloud_cognify(client, cognify_response)
            t_cognify = time.time() - t_cognify_start
        except Exception as e:
            t_cognify = time.time() - t_cognify_start
            status["cognify"] = f"failed: {_err(e)}"
            print(f"  Cognify FAILED: {_err(e)}")

        # ── Phase 3: search ──────────────────────────────────────────────
        # One timing per benchmarked search type, matching the local mode.
        # Scoped to this suite's dataset so a concurrently-running suite on
        # the same tenant cannot contaminate the search timing or results.
        if not _ingest_succeeded(status):
            _skip_search_phase(status, t_search)
        else:
            print("\nPhase 3: Running remote search queries...")
            for metric_key, search_type in BENCHMARKED_SEARCH_TYPES:
                t_q_start = time.time()
                try:
                    await client.search(
                        SEARCH_QUERY,
                        search_type=search_type,
                        datasets=[dataset_name],
                        only_context=True,
                    )
                    t_search[metric_key] = time.time() - t_q_start
                except Exception as e:
                    t_search[metric_key] = time.time() - t_q_start
                    status[f"search_{metric_key}"] = f"failed: {_err(e)}"
                    print(f"  Search {search_type} FAILED: {_err(e)}")

        # ── Phase 4 (cloud-only metric): delete the POPULATED dataset ────
        # Only meaningful with a graph in it, hence after cognify/search and
        # before tenant teardown.
        if config.get("create_tenant"):
            status["dataset_delete"] = "success"
            print(f"\nPhase 4: Deleting populated dataset '{dataset_name}'...")
            t_dataset_delete_start = time.time()
            try:
                await client.forget(dataset=dataset_name)
                t_dataset_delete = time.time() - t_dataset_delete_start
                print(f"  Dataset deleted in {t_dataset_delete:.2f}s")
            except Exception as e:
                t_dataset_delete = time.time() - t_dataset_delete_start
                status["dataset_delete"] = f"failed: {_err(e)}"
                print(f"  Dataset deletion FAILED: {_err(e)}")

        await client.close()

    t_total = t_add + t_cognify

    # ── Tenant teardown (only for tenants this run created) ──────────────
    if tenant_id is not None:
        print(f"\nDeleting benchmark tenant {tenant_id}...")
        t_tenant_delete_start = time.time()
        try:
            t_tenant_delete = await _delete_cloud_tenant(
                config["management_url"], config["tenant_api_key"], tenant_id
            )
            print(f"  Tenant deleted in {t_tenant_delete:.2f}s")
        except Exception as e:
            t_tenant_delete = time.time() - t_tenant_delete_start
            status["tenant_delete"] = f"failed: {_err(e)}"
            print(f"  Tenant deletion FAILED (manual cleanup needed for {tenant_id}): {e}")

    all_ok = all(v in ("success", "skipped") for v in status.values())

    # ── Report ───────────────────────────────────────────────────────────
    results = {
        "memories_count": n,
        "add_time_s": round(t_add, 3),
        "cognify_time_s": round(t_cognify, 3),
        "total_ingest_time_s": round(t_total, 3),
        "prune_time_s": round(t_prune, 3),
        "db_setup_time_s": 0.0,
        # A skipped search has no timing. Omit it rather than write 0.0 --
        # the file's own rule is elapsed-until-failure, never a fabricated
        # zero, and the report tolerates a missing metric.
        **{
            f"search_time_{key}": t_search[key]
            for key, _ in BENCHMARKED_SEARCH_TYPES
            if t_search.get(key) is not None
        },
        "status": status,
        "success": all_ok,
        "config": {
            "llm_model": "cloud (server-side)",
            "embedding_model": "cloud (server-side)",
            "embedding_dimensions": "server",
            "dataset_name": dataset_name,
            "mock_llm": False,
            "tenant_url": tenant_url,
            "created_tenant": bool(config.get("create_tenant")),
        },
    }
    if config.get("create_tenant"):
        results["tenant_create_time_s"] = round(t_tenant_create, 3)
        results["dataset_delete_time_s"] = round(t_dataset_delete, 3)
        results["tenant_delete_time_s"] = round(t_tenant_delete, 3)
        # Pre-run prune is skipped on a fresh tenant; drop the meaningless
        # zero so the report only shows metrics that were actually measured.
        results.pop("prune_time_s", None)

    print("\n" + "=" * 60)
    print("RESULTS (cloud)")
    print("=" * 60)
    print(f"  Tenant            : {tenant_url}")
    if config.get("create_tenant"):
        print(f"  Tenant create     : {t_tenant_create:.2f}s  [{status['tenant_create']}]")
        print(
            f"  Dataset delete    : {t_dataset_delete:.2f}s  "
            f"[{status.get('dataset_delete', 'skipped')}]"
        )
        print(f"  Tenant delete     : {t_tenant_delete:.2f}s  [{status['tenant_delete']}]")
    print(f"  Memories inserted : {n}")
    print(f"  add time          : {t_add:.2f}s  ({t_add / n:.2f}s per memory)  [{status['add']}]")
    print(f"  cognify time      : {t_cognify:.2f}s  [{status['cognify']}]")
    print(f"  Total ingest time : {t_total:.2f}s  ({t_total / n:.2f}s per memory)")
    for metric_key, search_type in BENCHMARKED_SEARCH_TYPES:
        timing = t_search.get(metric_key)
        shown = f"{timing:.2f}s" if timing is not None else "n/a  "
        print(f"  Search {search_type:<18}: {shown}  [{status[f'search_{metric_key}']}]")
    print(f"  Prune time        : {t_prune:.2f}s  [{status['prune']}]")
    print(f"  Overall           : {'ALL OK' if all_ok else 'SOME FAILURES'}")
    print("=" * 60)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Cognee memory ingestion (add + cognify).",
    )
    parser.add_argument(
        "--memories",
        type=Path,
        default=DEFAULT_MEMORIES_FILE,
        help=f"JSON file with memories array (default: {DEFAULT_MEMORIES_FILE.name})",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=f"LLM model (default: .env LLM_MODEL or {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"Embedding model (default: .env EMBEDDING_MODEL or {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help=f"LLM provider (default: .env LLM_PROVIDER or {DEFAULT_LLM_PROVIDER})",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help=f"Embedding provider (default: .env EMBEDDING_PROVIDER or {DEFAULT_EMBEDDING_PROVIDER})",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        default=None,
        help=f"Embedding dimensions (default: .env EMBEDDING_DIMENSIONS or {DEFAULT_EMBEDDING_DIMS})",
    )
    parser.add_argument(
        "--num-memories",
        type=int,
        default=None,
        help="Limit the number of memories to load (default: all)",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        default=False,
        help="Use mock LLM/embedding responses from mock_memories.json instead of real API calls",
    )
    parser.add_argument(
        "--mock-memories",
        type=Path,
        default=DEFAULT_MOCK_MEMORIES_FILE,
        help=f"Mock responses JSON file (default: {DEFAULT_MOCK_MEMORIES_FILE.name})",
    )
    parser.add_argument(
        "--mock-document-embeddings",
        type=Path,
        default=None,
        help=(
            "Embeddings file from capture_mock.py --capture-embeddings: replay "
            "the stored REAL document vectors; unseen texts (search queries) "
            "embed live, so a real embedding API key is still required."
        ),
    )
    parser.add_argument(
        "--tenant-url",
        default=None,
        help="Cognee Cloud tenant URL; runs all operations remotely via cognee.serve()",
    )
    parser.add_argument(
        "--tenant-api-key",
        default=None,
        help="API key for the cloud tenant (or set COGNEE_API_KEY)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Dataset name for cloud mode (default: bench_memories). Give each "
            "suite sharing a tenant its own name — cloud cleanup is dataset-scoped."
        ),
    )
    parser.add_argument(
        "--create-tenant",
        action="store_true",
        default=False,
        help=(
            "Cloud mode: create a fresh tenant via the tenant-controller API, "
            "measure creation time as its own metric, run the whole benchmark "
            "on it, and delete it afterwards. Replaces --tenant-url."
        ),
    )
    parser.add_argument(
        "--management-url",
        default="https://api.aws.cognee.ai",
        help="Tenant-controller API base URL for --create-tenant (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write JSON results to this file",
    )
    args = parser.parse_args()

    cloud_mode = bool(args.tenant_url or args.create_tenant)
    if cloud_mode:
        if args.mock_document_embeddings:
            sys.exit(
                "Error: --mock-document-embeddings is not supported in cloud mode "
                "(embeddings run server-side)"
            )
        config = _resolve_cloud_config(args)
    else:
        config = _resolve_config(args)
        if config["mock_llm"]:
            config["mock_memories_file"] = args.mock_memories
        if args.mock_document_embeddings:
            config["mock_document_embeddings_file"] = args.mock_document_embeddings

    memories = load_memories(args.memories)
    if args.num_memories is not None:
        memories = memories[: args.num_memories]
    print(f"Loaded {len(memories)} memories from {args.memories}")

    if cloud_mode:
        tenant_label = config["tenant_url"] or f"fresh tenant via {config['management_url']}"
        print(f"Config: cloud tenant {tenant_label}\n")
        results = asyncio.run(run_benchmark_cloud(memories, config=config))
    else:
        mock_label = " [MOCK]" if config["mock_llm"] else ""
        print(
            f"Config: llm={config['llm_model']}, embeddings={config['embedding_model']} ({config['embedding_dims']}d){mock_label}\n"
        )
        results = asyncio.run(run_benchmark(memories, config=config))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    # Propagate failures via the exit code so CI can gate on run outcomes.
    # The results JSON is already written above — orchestrators (the
    # percentile report) still collect the failed run's data.
    if not results.get("success", True):
        sys.exit(1)


if __name__ == "__main__":
    main()

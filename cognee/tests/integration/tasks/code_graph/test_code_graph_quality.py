"""Architecture queries over real source ingested through add -> cognify."""

import pytest

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info
from cognee.modules.retrieval.code_retriever import CodeRetriever


@pytest.mark.asyncio
async def test_component_hook_and_module_relationships_survive_ingestion(code_environment):
    env = code_environment
    source = env.root / "storefront"
    files = {
        "package.json": '{"name":"storefront","dependencies":{"react":"19.0.0"}}',
        "src/OrderCard.tsx": (
            'import { useOrder } from "../hooks/useOrder";\n'
            "export function OrderCard() { const order = useOrder(); "
            "return <div>{order.name}</div>; }\n"
        ),
        "hooks/useOrder.ts": (
            'import { useState } from "react";\n'
            'export function useOrder() { const [order] = useState({name: "demo"}); '
            "return order; }\n"
        ),
        # Ordinary utility symbols remain outside the component/hook view.
        "src/util.ts": 'export function formatLabel() { return "label"; }\n',
    }
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    added = await cognee.add(
        str(source),
        dataset_name="architecture",
        user=env.user,
        codegraph_config={"include_documents": False},
    )
    assert get_errored_run_info(added) is None
    runs = await cognee.cognify(
        datasets=[added.dataset_id],
        data_ids=added.data_ids,
        user=env.user,
        incremental_loading=False,
        data_cache=False,
    )
    assert runs and all(get_errored_run_info(run) is None for run in runs.values())

    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        overview = await CodeRetriever(config={"operation": "architecture"}).get_retrieved_objects(
            ""
        )
        details = await CodeRetriever(
            config={"operation": "architecture", "symbol_roles": ["component", "hook"]}
        ).get_retrieved_objects("")

    assert all(node["kind"] != "symbol" for node in overview["nodes"])
    assert any(
        edge["source"] == "src" and edge["target"] == "hooks" and edge["type"] == "calls"
        for edge in overview["edges"]
    )
    symbols = {node["name"]: node for node in details["nodes"] if node["kind"] == "symbol"}
    assert set(symbols) == {"src.OrderCard", "hooks.useOrder"}
    assert symbols["src.OrderCard"]["properties"]["web_component"] == "component"
    assert symbols["hooks.useOrder"]["properties"]["web_component"] == "hook"
    assert symbols["src.OrderCard"]["file"] == "src/OrderCard.tsx"
    assert symbols["hooks.useOrder"]["file"] == "hooks/useOrder.ts"
    assert all(node["repo"] == "storefront" and node["line"] == 2 for node in symbols.values())
    assert any(
        edge["source_id"] == symbols["src.OrderCard"]["id"]
        and edge["target_id"] == symbols["hooks.useOrder"]["id"]
        and edge["type"] == "calls"
        for edge in details["edges"]
    )
    assert "OrderCard" in details["diagram"]["source"]
    assert "useOrder" in details["diagram"]["source"]


@pytest.mark.asyncio
async def test_same_named_symbols_migrate_in_existing_dataset(code_environment):
    from uuid import UUID

    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.tasks.code_graph.extract_code_graph import fact_node_id
    from cognee.tasks.code_graph.models import CodeRepository, CodeSymbol

    env = code_environment
    source = env.root / "collision"
    (source / "src").mkdir(parents=True)
    (source / "package.json").write_text('{"name":"collision"}')
    for file, value in (("a.ts", 1), ("b.ts", 2)):
        (source / "src" / file).write_text(f"export function save() {{ return {value}; }}\n")
    added = await cognee.add(
        str(source),
        dataset_name="migration",
        user=env.user,
        codegraph_config={"include_documents": False},
    )
    assert get_errored_run_info(added) is None

    async def cognify(cache=False):
        runs = await cognee.cognify(
            datasets=[added.dataset_id],
            data_ids=added.data_ids,
            user=env.user,
            incremental_loading=cache,
            data_cache=cache,
        )
        assert runs and all(get_errored_run_info(run) is None for run in runs.values())

    await cognify(cache=True)
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        graph = await get_graph_engine()
        nodes, _ = await graph.get_graph_data()
        symbols = {str(key): props for key, props in nodes if props.get("type") == "CodeSymbol"}
        assert {props["file_path"] for props in symbols.values()} == {"src/a.ts", "src/b.ts"}
        assert len(symbols) == 2
        repository_id, repository = next(
            (key, props) for key, props in nodes if props.get("type") == "CodeRepository"
        )
        # Reproduce mapping v2: one symbol retains the first file's metadata.
        legacy_id = fact_node_id("collision", "symbol", "src.save")
        await graph.add_node(
            CodeSymbol(
                id=legacy_id, name="src.save", kind="symbol", repo="collision", file_path="src/a.ts"
            )
        )
        await graph.delete_nodes(list(symbols))
        await graph.add_node(
            CodeRepository(
                id=UUID(str(repository_id)),
                name="collision",
                path=str(source),
                last_snapshot_id=repository["last_snapshot_id"],
                last_mapping_version=2,
            )
        )

    from unittest.mock import MagicMock, patch

    with patch("cognee.tasks.code_graph.mapping_version.logger", MagicMock()) as log:
        await cognee.cognify(
            datasets=[added.dataset_id], user=env.user, incremental_loading=True, data_cache=True
        )
        log.warning.assert_called_once()
        assert "incremental_loading=False, data_cache=False" in log.warning.call_args.args[0]
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
        assert str(legacy_id) in {str(key) for key, _ in nodes}

    # An old merged symbol cannot safely choose which file a manual link meant.
    from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
    from cognee.infrastructure.engine.models.DataPoint import DataPoint
    from cognee.tasks.code_graph.symbol_migration import SymbolMigrationError

    class ReviewNote(DataPoint):
        text: str

    note = ReviewNote(text="Review save before release")
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        graph = await get_graph_engine()
        await graph.add_nodes([note])
        await graph.add_edges(
            [(str(note.id), str(legacy_id), "describes", {"evidence": note.text})]
        )
        # The earlier mapper's own edges carry no producer; they are not references.
        module_id = next(
            key
            for key, props in (await graph.get_graph_data())[0]
            if props.get("type") == "CodeModule"
        )
        await graph.add_edges([(str(module_id), str(legacy_id), "declares", {})])
        before_nodes, before_edges = await graph.get_graph_data()
    with pytest.raises(SymbolMigrationError, match="several file-scoped successors"):
        await cognify()
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        graph = await get_graph_engine()
        after_nodes, after_edges = await graph.get_graph_data()
        assert sorted(before_nodes, key=str) == sorted(after_nodes, key=str)
        assert sorted(before_edges, key=str) == sorted(after_edges, key=str)
        # The test explicitly removes its unresolved annotation before retrying.
        await graph.delete_edge_triples([EdgeIdentity(str(note.id), str(legacy_id), "describes")])
        await graph.delete_nodes([str(note.id)])

    await cognify()  # Same snapshot, same dataset; the mapping version forces rebuilding.
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        nodes, edges = await (await get_graph_engine()).get_graph_data()
        migrated = {str(key): props for key, props in nodes if props.get("type") == "CodeSymbol"}
        assert migrated.keys() == symbols.keys()
        assert str(legacy_id) not in {str(key) for key, _ in nodes}
        assert all(
            any(str(a) == key and kind == "declares" for a, _, kind, _ in edges) for key in migrated
        )
        assert all(
            props.get("producer") == "cognee_code_graph"
            for _, _, kind, props in edges
            if kind == "declares"
        )
    await cognify()  # Retry keeps the split identities and edge count stable.
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        repeated_nodes, repeated_edges = await (await get_graph_engine()).get_graph_data()
        assert {str(key) for key, _ in repeated_nodes} == {str(key) for key, _ in nodes}
        assert len(repeated_edges) == len(edges)


@pytest.mark.asyncio
async def test_external_references_survive_symbol_migration_and_retry(
    code_environment, monkeypatch
):
    import importlib
    from types import SimpleNamespace
    from uuid import uuid4

    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
    from cognee.infrastructure.databases.provenance.source_refs import (
        make_source_ref_key,
        make_source_run_ref,
    )
    from cognee.infrastructure.engine.models.DataPoint import DataPoint
    from cognee.modules.pipelines.models import PipelineContext
    from cognee.tasks.code_graph.extract_code_graph import fact_node_id
    from cognee.tasks.code_graph.link_code_references import CodeReference, link_code_references
    from cognee.tasks.code_graph.models import CodeRepository, CodeSymbol

    class SourceDocument(DataPoint):
        text: str

    env = code_environment
    root = env.root / "references"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="references"\nversion="0.1.0"\n')
    (root / "main.py").write_text("def save():\n    return 1\n")
    added = await cognee.add(
        str(root),
        dataset_name="references",
        user=env.user,
        codegraph_config={"include_documents": False},
    )

    async def ingest():
        runs = await cognee.cognify(
            datasets=[added.dataset_id],
            data_ids=added.data_ids,
            user=env.user,
            incremental_loading=False,
            data_cache=False,
        )
        return [get_errored_run_info(run) for run in runs.values()]

    assert all(error is None for error in await ingest())
    document_data = await cognee.add(
        "SDK-42: main.py save must preserve orders", dataset_name="references", user=env.user
    )
    ctx = PipelineContext(
        user=env.user,
        dataset=SimpleNamespace(id=added.dataset_id),
        data_item=SimpleNamespace(id=document_data.data_ids[0]),
        pipeline_run_id=uuid4(),
    )
    ref_key = make_source_ref_key(added.dataset_id, document_data.data_ids[0])
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        graph = await get_graph_engine()
        nodes, _ = await graph.get_graph_data()
        repo_id, _repo = next(
            (key, props) for key, props in nodes if props.get("type") == "CodeRepository"
        )
        new_id, symbol = next(
            (str(key), props) for key, props in nodes if props.get("type") == "CodeSymbol"
        )
        source = SourceDocument(text="SDK-42: main.py save must preserve orders")
        await graph.add_nodes(
            [source], source_ref_key=ref_key, pipeline_run_id=str(ctx.pipeline_run_id)
        )
        reference = CodeReference(
            source_node_id=source.id,
            repo="references",
            file="main.py",
            symbol=symbol["name"],
            evidence=source.text,
            source_url="https://linear.app/example/issue/SDK-42",
            external_id="SDK-42",
        )
        result = await link_code_references([reference], ctx)
        assert result["linked"] == [{"source_id": str(source.id), "target_id": new_id}]
        assert (await link_code_references([reference], ctx)) == result
    from cognee.modules.data.models import Data
    from cognee.modules.pipelines import Task
    from examples.guides.code_reference_task import link_source

    pipeline_runs = await cognee.run_custom_pipeline(
        tasks=[Task(link_source, reference=reference)],
        data=[Data(id=document_data.data_ids[0])],
        dataset=added.dataset_id,
        user=env.user,
        pipeline_name="test_explicit_code_references",
        skip_connection_test=True,
    )
    assert all(get_errored_run_info(run) is None for run in pipeline_runs.values())

    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        graph = await get_graph_engine()
        missing = reference.model_copy(update={"repo": "other"})
        assert (await link_code_references([missing], ctx))["unresolved"][0][
            "reason"
        ] == "not_found"
        absent = reference.model_copy(update={"source_node_id": uuid4()})
        with pytest.raises(CogneeValidationError, match="absent"):
            await link_code_references([absent], ctx)
        wrong_ctx = PipelineContext(
            user=env.user,
            dataset=ctx.dataset,
            data_item=SimpleNamespace(id=uuid4()),
            pipeline_run_id=uuid4(),
        )
        with pytest.raises(CogneeValidationError, match="belong"):
            await link_code_references([reference], wrong_ctx)
        read = await CodeRetriever(
            config={"operation": "references", "id": new_id}
        ).get_retrieved_objects("")
        assert read["references"][0]["evidence"]["external_id"] == "SDK-42"
        # Recreate the old identity and an original document-owned edge.
        old_id = fact_node_id("references", "symbol", symbol["name"])
        await graph.add_nodes(
            [
                CodeSymbol(
                    id=old_id,
                    name=symbol["name"],
                    kind="symbol",
                    repo="references",
                    file_path="main.py",
                )
            ]
        )
        await graph.add_edges(
            [(str(source.id), str(old_id), "references_code", read["references"][0]["evidence"])],
            source_ref_key=ref_key,
            pipeline_run_id=str(ctx.pipeline_run_id),
        )
        await graph.delete_nodes([new_id])
        await graph.add_nodes(
            [CodeRepository(id=repo_id, name="references", path=str(root), last_mapping_version=2)]
        )
    # A failed ownership transfer must not delete the old symbol or stamp success.
    migration = importlib.import_module("cognee.tasks.code_graph.symbol_migration")
    original = migration.transfer_symbol_references
    extraction = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")

    async def fail_after_copy(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("simulated interruption before sweep")

    with monkeypatch.context() as patch:
        patch.setattr(extraction, "transfer_symbol_references", fail_after_copy)
        from cognee.modules.pipelines.exceptions.exceptions import CognifyFailedError

        with pytest.raises(CognifyFailedError, match="simulated interruption"):
            await ingest()
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
        assert str(old_id) in {str(key) for key, _ in nodes}
    assert all(error is None for error in await ingest())
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        graph = await get_graph_engine()
        nodes, edges = await graph.get_graph_data()
        assert str(old_id) not in {str(key) for key, _ in nodes}
        matches = [
            (s, t, r, p)
            for s, t, r, p in edges
            if str(s) == str(source.id) and r == "references_code"
        ]
        assert len(matches) == 1 and str(matches[0][1]) == new_id
        assert matches[0][3]["evidence"] == source.text
        key = EdgeIdentity(str(source.id), new_id, "references_code")
        owner = (await graph.get_edge_delete_data([key]))[key]
        assert ref_key in owner.source_ref_keys
        assert make_source_run_ref(ctx.pipeline_run_id, ref_key) in owner.source_run_refs


@pytest.mark.asyncio
async def test_impact_paths_from_real_frontend_and_backend_sources(code_environment):
    import shutil
    from pathlib import Path

    env = code_environment
    fixture = Path(__file__).parent / "fixtures" / "impact"
    dataset_id = None
    for name in ("frontend", "backend"):
        source = env.root / name
        shutil.copytree(fixture / name, source)
        added = await cognee.add(
            str(source),
            dataset_name="impact",
            user=env.user,
            codegraph_config={"include_documents": False},
        )
        runs = await cognee.cognify(
            datasets=[added.dataset_id],
            data_ids=added.data_ids,
            user=env.user,
            incremental_loading=False,
            data_cache=False,
        )
        assert all(get_errored_run_info(run) is None for run in runs.values())
        dataset_id = added.dataset_id
    async with set_database_global_context_variables(dataset_id, env.user.id):

        async def query(**config):
            return await CodeRetriever(config=config).get_retrieved_objects("")

        frontend = await query(
            operation="find_path",
            repo="frontend",
            target_repo="frontend",
            **{"from": "src.OrderCard", "to": "/orders"},
        )
        assert frontend["found"]
        assert [node["name"] for node in frontend["path"]] == [
            "src.OrderCard",
            "hooks.useOrder",
            "/orders",
        ]
        backend = await query(
            operation="find_path", repo="backend", **{"from": "/orders", "to": "api.load_orders"}
        )
        assert backend["found"]
        assert [node["name"] for node in backend["path"]] == [
            "/orders",
            "api.list_orders",
            "api.load_orders",
        ]
        impact = await query(
            operation="impact_analysis",
            repo="frontend",
            target="hooks.useOrder",
            relation_types=["calls"],
        )
        assert "src.OrderCard" in {
            node["name"] for values in impact["by_depth"].values() for node in values
        }
        # A literal URL shared across repos does not prove deployment wiring.
        cross = await query(
            operation="find_path", **{"from": "src.OrderCard", "to": "api.list_orders"}
        )
        assert not cross["found"]
        storage = await query(
            operation="find_path", repo="backend", **{"from": "api.load_orders", "to": "orders"}
        )
        assert not storage["found"]  # Enola 0.4.30 does not emit this ORM query edge.
        coverage = await query(operation="coverage")
        assert {report["repo"] for report in coverage["repositories"]} == {"frontend", "backend"}
        assert all(report["mapping"] is not None for report in coverage["repositories"])


@pytest.mark.asyncio
async def test_relational_migration_preserves_owner_and_other_dataset(code_environment):
    from uuid import uuid4

    from sqlalchemy import select

    from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.graph.methods.upsert_edges import upsert_edges
    from cognee.modules.graph.models.Edge import Edge
    from cognee.tasks.code_graph.symbol_migration import _transfer_ledger

    user = code_environment.user
    dataset, other_dataset, data, run = (uuid4() for _ in range(4))
    source, old, new = (uuid4() for _ in range(3))
    properties = {"evidence": "manual", "target_node_id": str(old)}
    for dataset_id in (dataset, other_dataset):
        await upsert_edges(
            [(source, old, "describes", properties)],
            tenant_id=user.tenant_id,
            user_id=user.id,
            data_id=data,
            dataset_id=dataset_id,
            pipeline_run_id=run,
        )
    pairs = [
        (
            EdgeIdentity(str(source), str(old), "describes"),
            EdgeIdentity(str(source), str(new), "describes"),
        )
    ]
    await _transfer_ledger(pairs, dataset)
    await _transfer_ledger(pairs, dataset)  # Idempotent after the source row was removed.
    async with get_relational_engine().get_async_session() as session:
        rows = (
            (await session.execute(select(Edge).where(Edge.source_node_id == source)))
            .scalars()
            .all()
        )
        by_dataset = {row.dataset_id: row for row in rows}
        assert set(by_dataset) == {dataset, other_dataset}
        migrated = by_dataset[dataset]
        assert migrated.destination_node_id == new
        assert (migrated.user_id, migrated.data_id, migrated.pipeline_run_id) == (
            user.id,
            data,
            run,
        )
        assert migrated.attributes == {"evidence": "manual", "target_node_id": str(new)}
        assert by_dataset[other_dataset].destination_node_id == old


@pytest.mark.asyncio
async def test_relational_migration_transfers_every_chunk(code_environment, monkeypatch):
    """A migration with more edges than one lookup chunk moves all of them."""
    from uuid import uuid4

    from sqlalchemy import select

    from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.graph.methods.upsert_edges import upsert_edges
    from cognee.modules.graph.models.Edge import Edge
    from cognee.tasks.code_graph import symbol_migration

    monkeypatch.setattr(symbol_migration, "_LEDGER_CHUNK_PAIRS", 1)
    user = code_environment.user
    dataset, data, run = uuid4(), uuid4(), uuid4()
    old, new = uuid4(), uuid4()
    sources = [uuid4() for _ in range(3)]
    for source in sources:
        await upsert_edges(
            [(source, old, "describes", {})],
            tenant_id=user.tenant_id,
            user_id=user.id,
            data_id=data,
            dataset_id=dataset,
            pipeline_run_id=run,
        )
    pairs = [
        (
            EdgeIdentity(str(source), str(old), "describes"),
            EdgeIdentity(str(source), str(new), "describes"),
        )
        for source in sources
    ]
    await symbol_migration._transfer_ledger(pairs, dataset)
    async with get_relational_engine().get_async_session() as session:
        rows = (
            (await session.execute(select(Edge).where(Edge.source_node_id.in_(sources))))
            .scalars()
            .all()
        )
    assert len(rows) == 3 and {row.destination_node_id for row in rows} == {new}


@pytest.mark.asyncio
async def test_loader_owned_call_is_removed_when_source_call_disappears(code_environment):
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.tasks.code_graph.symbol_migration import CODE_EDGE_OWNER

    env = code_environment
    root = env.root / "owned"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="owned"\nversion="0.1.0"\n')
    source = root / "main.py"
    source.write_text("def helper():\n    return 1\n\ndef caller():\n    return helper()\n")

    async def ingest():
        added = await cognee.add(
            str(root),
            dataset_name="ownership",
            user=env.user,
            codegraph_config={"include_documents": False},
        )
        runs = await cognee.cognify(
            datasets=[added.dataset_id],
            data_ids=added.data_ids,
            user=env.user,
            incremental_loading=False,
            data_cache=False,
        )
        assert all(get_errored_run_info(run) is None for run in runs.values())
        return added.dataset_id

    dataset_id = await ingest()
    async with set_database_global_context_variables(dataset_id, env.user.id):
        nodes, edges = await (await get_graph_engine()).get_graph_data()
        symbols = {str(key) for key, props in nodes if props.get("type") == "CodeSymbol"}
        calls = [(str(s), str(t), r, p) for s, t, r, p in edges if r == "calls"]
        assert len(calls) == 1 and calls[0][3]["producer"] == CODE_EDGE_OWNER
        old_edge = calls[0][:3]
    source.write_text("def helper():\n    return 1\n\ndef caller():\n    return 1\n")
    await ingest()
    async with set_database_global_context_variables(dataset_id, env.user.id):
        nodes, edges = await (await get_graph_engine()).get_graph_data()
        assert symbols == {str(key) for key, props in nodes if props.get("type") == "CodeSymbol"}
        assert old_edge not in {(str(s), str(t), r) for s, t, r, _ in edges}

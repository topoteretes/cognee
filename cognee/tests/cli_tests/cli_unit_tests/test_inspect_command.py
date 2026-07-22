import os
import sys
import json
import pytest
import argparse
import asyncio
from uuid import UUID
from unittest.mock import patch, MagicMock, AsyncMock

import cognee
from cognee.cli.commands.inspect_command import InspectCommand
from cognee.cli.exceptions import CliCommandException


# Mock asyncio.run to properly handle coroutines in synchronous test environment
def _mock_run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_inspect_command_properties():
    """Test command properties and configuration."""
    command = InspectCommand()
    assert command.command_string == "inspect"
    assert "Inspect stored memory" in command.help_string

    parser = argparse.ArgumentParser()
    command.configure_parser(parser)

    # Verify argparse arguments
    actions = {action.dest: action for action in parser._actions}
    assert "json" in actions
    assert "inspect_action" in actions


def test_inspect_command_execution(capsys):
    """Integration test for inspect commands against a seeded database."""
    # 1. Set environment variable to bypass LLM and embedding connection tests
    os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"

    # 2. Setup temporary directories for cognee config to isolate tests
    data_directory_path = os.path.join(
        os.path.dirname(__file__), "../../../.data_storage/test_inspect_command"
    )
    cognee_directory_path = os.path.join(
        os.path.dirname(__file__), "../../../.cognee_system/test_inspect_command"
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    # 3. Prune existing test files
    _mock_run(cognee.prune.prune_data())
    _mock_run(cognee.prune.prune_system(metadata=True))

    # 4. Setup schemas
    from cognee.modules.engine.operations.setup import setup

    _mock_run(setup())

    # 5. Seed data
    from cognee.modules.users.methods import get_default_user
    from cognee.modules.data.methods import create_dataset
    from cognee.modules.users.permissions.methods import give_permission_on_dataset
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.metrics import (
        ensure_and_touch_session,
        accumulate_usage,
        mark_ended,
        SessionStatus,
    )

    user = _mock_run(get_default_user())

    async def seed_data():
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            dataset = await create_dataset(
                dataset_name="test_inspect_dataset",
                user=user,
                session=session,
            )
            for perm in ("read", "write", "share", "delete"):
                await give_permission_on_dataset(user, dataset.id, perm)

        # Add raw items (this populates Data table)
        await cognee.add(
            data="Hello world from inspect CLI tests.",
            dataset_name="test_inspect_dataset",
            user=user,
        )

        # Seed a session
        session_id = "test_inspect_sess_123"
        await ensure_and_touch_session(
            session_id=session_id,
            user_id=user.id,
            dataset_id=dataset.id,
        )
        await accumulate_usage(
            session_id=session_id,
            user_id=user.id,
            tokens_in=50,
            tokens_out=150,
            cost_usd=0.002,
        )
        await mark_ended(
            session_id=session_id,
            user_id=user.id,
            status=SessionStatus.COMPLETED,
        )

    _mock_run(seed_data())

    command = InspectCommand()

    # Define a helper to run commands and capture print
    def run_cmd(args):
        # We patch sys.stdout to prevent capturing issues with standard capsys in some environments
        # but also use capsys for extra coverage.
        captured_output = []

        def mocked_write(data):
            if isinstance(data, bytes):
                captured_output.append(data.decode("utf-8"))
            else:
                captured_output.append(str(data))

        with patch("sys.stdout.write", new=mocked_write):
            with patch("cognee.cli.commands.inspect_command.asyncio.run", side_effect=_mock_run):
                command.execute(args)
        return "".join(captured_output)

    # Mock get_graph_metrics on graph engine to return deterministic numbers
    mock_metrics = {"num_nodes": 42, "num_edges": 128}
    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_metrics.return_value = mock_metrics

    # Test 1: overview (table format)
    with patch(
        "cognee.infrastructure.databases.graph.get_graph_engine", return_value=mock_graph_engine
    ):
        args_overview = argparse.Namespace(
            inspect_action="overview",
            json=False,
            user_id=str(user.id),
        )
        output = run_cmd(args_overview)
        assert "Memory Overview" in output
        assert "test_inspect_dataset" in output
        assert "Completed" in output

    # Test 2: overview (JSON format)
    with patch(
        "cognee.infrastructure.databases.graph.get_graph_engine", return_value=mock_graph_engine
    ):
        args_overview_json = argparse.Namespace(
            inspect_action="overview",
            json=True,
            user_id=str(user.id),
        )
        output_json = run_cmd(args_overview_json)
        parsed = json.loads(output_json)
        assert parsed["totals"]["datasets_count"] == 1
        assert parsed["totals"]["graph_nodes_count"] == 42
        assert parsed["totals"]["graph_edges_count"] == 128
        assert parsed["sessions_by_status"]["completed"] == 1

    # Test 3: dataset drill-down (table format)
    args_ds = argparse.Namespace(
        inspect_action="dataset",
        name_or_id="test_inspect_dataset",
        limit=None,
        json=False,
        user_id=str(user.id),
    )
    output_ds = run_cmd(args_ds)
    assert "Dataset: test_inspect_dataset" in output_ds
    assert "Documents" in output_ds

    # Test 4: dataset drill-down (JSON format)
    args_ds_json = argparse.Namespace(
        inspect_action="dataset",
        name_or_id="test_inspect_dataset",
        limit=None,
        json=True,
        user_id=str(user.id),
    )
    output_ds_json = run_cmd(args_ds_json)
    parsed_ds = json.loads(output_ds_json)
    assert parsed_ds["dataset_name"] == "test_inspect_dataset"
    assert len(parsed_ds["documents"]) == 1

    # Test 5: sessions (table format)
    args_sess = argparse.Namespace(
        inspect_action="sessions",
        limit=50,
        json=False,
        user_id=str(user.id),
    )
    output_sess = run_cmd(args_sess)
    assert "Conversation Sessions" in output_sess
    assert "test_inspect_sess_123" in output_sess

    # Test 6: sessions (JSON format)
    args_sess_json = argparse.Namespace(
        inspect_action="sessions",
        limit=50,
        json=True,
        user_id=str(user.id),
    )
    output_sess_json = run_cmd(args_sess_json)
    parsed_sess = json.loads(output_sess_json)
    assert len(parsed_sess) == 1
    assert parsed_sess[0]["effective_status"] == "completed"

    # Test 7: recent (table format)
    args_recent = argparse.Namespace(
        inspect_action="recent",
        limit=5,
        json=False,
        user_id=str(user.id),
    )
    output_recent = run_cmd(args_recent)
    assert "Recently Ingested Items" in output_recent
    assert "test_inspect_dataset" in output_recent

    # Test 8: recent (JSON format)
    args_recent_json = argparse.Namespace(
        inspect_action="recent",
        limit=5,
        json=True,
        user_id=str(user.id),
    )
    output_recent_json = run_cmd(args_recent_json)
    parsed_recent = json.loads(output_recent_json)
    assert len(parsed_recent) == 1
    assert parsed_recent[0]["dataset_name"] == "test_inspect_dataset"

    # Test 9: Access Control Overview for Unauthorized User
    from cognee.modules.users.methods import create_user
    async def create_unauth():
        return await create_user(
            email="unauthorized_inspect@example.com",
            password="password123",
        )
    unauthorized_user = _mock_run(create_unauth())

    with patch("cognee.infrastructure.databases.graph.get_graph_engine", return_value=mock_graph_engine):
        args_unauth_overview = argparse.Namespace(
            inspect_action="overview",
            json=True,
            user_id=str(unauthorized_user.id),
        )
        output_unauth = run_cmd(args_unauth_overview)
        parsed_unauth = json.loads(output_unauth)
        assert parsed_unauth["totals"]["datasets_count"] == 0
        assert parsed_unauth["totals"]["documents_count"] == 0
        assert len(parsed_unauth["datasets"]) == 0
        assert len(parsed_unauth["recent_ingests"]) == 0

    # Test 10: Access Control Dataset drill-down for Unauthorized User
    args_unauth_ds = argparse.Namespace(
        inspect_action="dataset",
        name_or_id="test_inspect_dataset",
        limit=None,
        json=False,
        user_id=str(unauthorized_user.id),
    )
    with pytest.raises(CliCommandException) as exc_info:
        run_cmd(args_unauth_ds)
    assert "not found or not accessible" in str(exc_info.value)

    # Clean up environment variable
    del os.environ["COGNEE_SKIP_CONNECTION_TEST"]

    # Clean up test directories
    _mock_run(cognee.prune.prune_data())
    _mock_run(cognee.prune.prune_system(metadata=True))

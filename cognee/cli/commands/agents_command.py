import argparse
import asyncio
import json
from uuid import UUID

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


class AgentsCommand(SupportsCliCommand):
    command_string = "agents"
    help_string = "Manage agents (create, list, get, delete, register, unregister, connections)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Manage Cognee agents and their connections.

Subcommands:
  create        Create a new agent and grant it access to datasets
  list          List all agents you own
  get           Show details for a single agent
  delete        Delete an agent by ID
  register      Register an agent connection (session)
  unregister    Unregister an agent connection (session)
  connections   List active agent connections and their memory sources
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="agents_action", title="actions")

        # create
        p_create = sub.add_parser("create", help="Create a new agent")
        p_create.add_argument("name", help="Agent name")
        p_create.add_argument(
            "--datasets",
            nargs="*",
            default=None,
            help="Dataset names or UUIDs to grant the agent read/write access to",
        )

        # list
        sub.add_parser("list", help="List all agents you own")

        # get
        p_get = sub.add_parser("get", help="Show details for a single agent")
        p_get.add_argument("agent_id", help="Agent UUID")

        # delete
        p_del = sub.add_parser("delete", help="Delete an agent by ID")
        p_del.add_argument("agent_id", help="Agent UUID")
        p_del.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

        # register
        p_reg = sub.add_parser("register", help="Register an agent connection")
        p_reg.add_argument("agent_session_name", help="Agent session name")
        p_reg.add_argument("--type", default="api", help="Connection type (default: api)")
        p_reg.add_argument(
            "--memory-mode",
            dest="memory_mode",
            default="unknown",
            help="Memory mode (default: unknown)",
        )
        p_reg.add_argument(
            "--session-id",
            dest="session_id",
            default=None,
            help="Optional session id",
        )
        p_reg.add_argument(
            "--dataset-ids",
            dest="dataset_ids",
            nargs="*",
            default=None,
            help="Dataset UUIDs to associate with the connection",
        )
        p_reg.add_argument(
            "--dataset-names",
            dest="dataset_names",
            nargs="*",
            default=None,
            help="Dataset names to associate with the connection",
        )

        # unregister
        p_unreg = sub.add_parser("unregister", help="Unregister an agent connection")
        p_unreg.add_argument("agent_session_name", help="Agent session name")

        # connections
        p_conn = sub.add_parser("connections", help="List active agent connections")
        p_conn.add_argument(
            "--agent-id",
            dest="agent_id",
            default=None,
            help="Filter by agent UUID",
        )
        p_conn.add_argument(
            "--range",
            dest="range_key",
            default="30d",
            help="Time range key (default: 30d)",
        )
        p_conn.add_argument(
            "--status",
            dest="status_filter",
            default=None,
            help="Filter by connection status",
        )
        p_conn.add_argument("--limit", type=int, default=50, help="Max results (default: 50)")
        p_conn.add_argument("--offset", type=int, default=0, help="Result offset (default: 0)")

    def execute(self, args: argparse.Namespace) -> None:
        action = getattr(args, "agents_action", None)
        if not action:
            fmt.error("No action specified. Use --help to see available actions.")
            raise CliCommandException("No action specified", error_code=1)

        dispatch = {
            "create": self._create,
            "list": self._list,
            "get": self._get,
            "delete": self._delete,
            "register": self._register,
            "unregister": self._unregister,
            "connections": self._connections,
        }
        dispatch[action](args)

    def _create(self, args: argparse.Namespace) -> None:
        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            result = await cognee.agents.create(args.name, datasets=args.datasets, user=user)

            fmt.success(f"Created agent '{args.name}' ({result['agent_id']})")
            fmt.echo(f"Agent ID:    {result['agent_id']}")
            fmt.echo(f"Agent email: {result['agent_email']}")
            fmt.echo(f"API key:     {result['agent_api_key']}")
            fmt.warning(
                "Store this API key now. It is a one-time secret and cannot be retrieved again."
            )

        asyncio.run(run())

    def _list(self, args: argparse.Namespace) -> None:
        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            agents = await cognee.agents.list(user=user)
            if not agents:
                fmt.echo("No agents found.")
                return
            fmt.echo(f"{'Agent ID':<38} {'Agent Email':<40} {'API Key Label'}")
            fmt.echo("-" * 100)
            for a in agents:
                fmt.echo(f"{str(a['agent_id']):<38} {a['agent_email']:<40} {a['api_key_label']}")

        asyncio.run(run())

    def _get(self, args: argparse.Namespace) -> None:
        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            info = await cognee.agents.get(args.agent_id, user=user)
            fmt.echo(f"Agent ID:      {info['agent_id']}")
            fmt.echo(f"Agent email:   {info['agent_email']}")
            fmt.echo(f"API key label: {info['api_key_label']}")

        asyncio.run(run())

    def _delete(self, args: argparse.Namespace) -> None:
        agent_id = args.agent_id
        if not args.force:
            if not fmt.confirm(f"Delete agent {agent_id}? This cannot be undone"):
                fmt.echo("Cancelled.")
                return

        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            await cognee.agents.delete(agent_id, user=user)
            fmt.success(f"Agent {agent_id} deleted.")

        asyncio.run(run())

    def _register(self, args: argparse.Namespace) -> None:
        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            connection = await cognee.agents.register(
                args.agent_session_name,
                user=user,
                type=args.type,
                memory_mode=args.memory_mode,
                session_id=args.session_id,
                dataset_ids=args.dataset_ids,
                dataset_names=args.dataset_names,
            )
            connection_id = connection.get("id", args.agent_session_name)
            fmt.success(f"Registered agent connection {connection_id}")
            fmt.echo(json.dumps(connection, indent=2, default=str))

        asyncio.run(run())

    def _unregister(self, args: argparse.Namespace) -> None:
        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            count = await cognee.agents.unregister(args.agent_session_name, user=user)
            fmt.success(f"Active connections: {count}")

        asyncio.run(run())

    def _connections(self, args: argparse.Namespace) -> None:
        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            agent_id = UUID(args.agent_id) if args.agent_id else None
            response = await cognee.agents.list_connections(
                user=user,
                agent_id=agent_id,
                range_key=args.range_key,
                status_filter=args.status_filter,
                limit=args.limit,
                offset=args.offset,
            )

            agents = response.get("agents", []) or []
            if not agents:
                fmt.echo("No agent connections found.")
            else:
                fmt.echo(f"{'Agent ID':<38} {'Session':<30} {'Status':<12} {'Last Active'}")
                fmt.echo("-" * 100)
                for a in agents:
                    fmt.echo(
                        f"{str(a.get('agent_id', '')):<38} "
                        f"{str(a.get('agent_session_name', '')):<30} "
                        f"{str(a.get('status', '')):<12} "
                        f"{a.get('last_active_at', '')}"
                    )

            memory_sources = response.get("memory_sources")
            if memory_sources:
                fmt.echo("")
                fmt.echo("Memory sources:")
                fmt.echo(json.dumps(memory_sources, indent=2, default=str))

        asyncio.run(run())

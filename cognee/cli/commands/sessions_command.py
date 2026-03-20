import argparse
import asyncio
import json

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


class SessionsCommand(SupportsCliCommand):
    command_string = "sessions"
    help_string = "View conversation sessions and Q&A history"
    docs_url = DEFAULT_DOCS_URL
    description = """
View and manage Cognee conversation sessions.

Subcommands:
  get       Retrieve Q&A entries from a session
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="sessions_action", title="actions")

        p_get = sub.add_parser("get", help="Retrieve session Q&A history")
        p_get.add_argument(
            "session_id",
            nargs="?",
            default=None,
            help="Session ID (default: scoped to the current --user-id)",
        )
        p_get.add_argument(
            "-n",
            "--last-n",
            type=int,
            default=None,
            help="Return only the last N entries",
        )
        p_get.add_argument(
            "-f",
            "--format",
            choices=["pretty", "json"],
            default="pretty",
            dest="output_format",
            help="Output format (default: pretty)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        action = getattr(args, "sessions_action", None)
        if not action:
            fmt.error("No action specified. Use --help to see available actions.")
            raise CliCommandException("No action specified", error_code=1)

        if action == "get":
            self._get(args)

    def _get(self, args: argparse.Namespace) -> None:
        async def run():
            from cognee.api.v1.session import get_session
            from cognee.cli.user_resolution import resolve_cli_user, scoped_session_id

            user = await resolve_cli_user(getattr(args, "user_id", None))
            sid = scoped_session_id(user.id, args.session_id)

            entries = await get_session(
                session_id=sid,
                last_n=args.last_n,
                user=user,
            )
            if not entries:
                fmt.echo(f"No entries in session '{sid}'.")
                return

            if args.output_format == "json":
                fmt.echo(
                    json.dumps(
                        [e.model_dump(mode="json") for e in entries],
                        indent=2,
                        default=str,
                    )
                )
                return

            for entry in entries:
                fmt.echo(f"[{entry.qa_id or 'no-id'}]")
                if hasattr(entry, "question") and entry.question:
                    fmt.echo(f"  Q: {entry.question}")
                if hasattr(entry, "answer") and entry.answer:
                    fmt.echo(f"  A: {entry.answer}")
                if hasattr(entry, "feedback_score") and entry.feedback_score is not None:
                    fmt.echo(f"  Score: {entry.feedback_score}")
                if hasattr(entry, "feedback_text") and entry.feedback_text:
                    fmt.echo(f"  Feedback: {entry.feedback_text}")
                fmt.echo("")

        asyncio.run(run())

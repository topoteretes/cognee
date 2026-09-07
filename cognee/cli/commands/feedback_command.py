import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


class FeedbackCommand(SupportsCliCommand):
    command_string = "feedback"
    help_string = "Add or remove feedback on session Q&A entries"
    docs_url = DEFAULT_DOCS_URL
    description = """
Manage feedback on Cognee session Q&A entries.

Subcommands:
  add       Attach feedback (text and/or score) to a Q&A entry
  delete    Clear feedback from a Q&A entry
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="feedback_action", title="actions")

        # add
        p_add = sub.add_parser("add", help="Add feedback to a Q&A entry")
        p_add.add_argument("session_id", help="Session ID")
        p_add.add_argument("qa_id", help="Q&A entry ID")
        p_add.add_argument("-t", "--text", default=None, help="Feedback text")
        p_add.add_argument("-s", "--score", type=int, default=None, help="Feedback score (integer)")

        # delete
        p_del = sub.add_parser("delete", help="Clear feedback from a Q&A entry")
        p_del.add_argument("session_id", help="Session ID")
        p_del.add_argument("qa_id", help="Q&A entry ID")

    def execute(self, args: argparse.Namespace) -> None:
        action = getattr(args, "feedback_action", None)
        if not action:
            fmt.error("No action specified. Use --help to see available actions.")
            raise CliCommandException("No action specified", error_code=1)

        dispatch = {
            "add": self._add,
            "delete": self._delete,
        }
        dispatch[action](args)

    def _add(self, args: argparse.Namespace) -> None:
        if args.text is None and args.score is None:
            fmt.error("Provide at least --text or --score.")
            raise CliCommandException("No feedback content provided", error_code=1)

        sid = args.session_id

        async def run() -> bool:
            from cognee.api.v1.session import add_feedback
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            return await add_feedback(
                session_id=sid,
                qa_id=args.qa_id,
                feedback_text=args.text,
                feedback_score=args.score,
                user=user,
            )

        ok = self._run(run(), "Error adding feedback")
        if not ok:
            raise CliCommandException(
                _not_found_message("Feedback not added", args.qa_id, sid), error_code=1
            )
        fmt.success(f"Feedback added to entry {args.qa_id} in session {sid}.")

    def _delete(self, args: argparse.Namespace) -> None:
        sid = args.session_id

        async def run() -> bool:
            from cognee.api.v1.session import delete_feedback
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            return await delete_feedback(
                session_id=sid,
                qa_id=args.qa_id,
                user=user,
            )

        ok = self._run(run(), "Error clearing feedback")
        if not ok:
            raise CliCommandException(
                _not_found_message("Feedback not cleared", args.qa_id, sid), error_code=1
            )
        fmt.success(f"Feedback cleared from entry {args.qa_id} in session {sid}.")

    @staticmethod
    def _run(coro, error_prefix: str) -> bool:
        """Run the session call, surfacing infrastructure failures as CLI errors.

        The session API raises on anything that is not a plain "not found"
        (unreachable cache, misconfigured backend, invalid ids), so a failure
        here is a real error and must not be reported as a missing entry.
        """
        try:
            return asyncio.run(coro)
        except Exception as e:
            raise CliCommandException(f"{error_prefix}: {str(e)}", error_code=1) from e


def _not_found_message(prefix: str, qa_id: str, session_id: str) -> str:
    return (
        f"{prefix}: no Q&A entry {qa_id} in session {session_id}, or caching is disabled. "
        "Check the session and qa IDs and the CACHING setting."
    )

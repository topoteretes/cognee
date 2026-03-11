import argparse
import asyncio
import json
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class SkillsCommand(SupportsCliCommand):
    command_string = "skills"
    help_string = "Manage and query cognee-skills: ingest, recommend, list, observe, promote"
    docs_url = DEFAULT_DOCS_URL
    description = """
Manage and query cognee-skills for intelligent skill routing.

Sub-commands:

  ingest      Parse SKILL.md files from a folder and store in the knowledge graph
  recommend   Find the best skills for a task using semantic search + learned preferences
  list        List all ingested skills
  observe     Record a skill execution outcome
  promote     Promote cached runs to the graph and update preference weights
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="skills_action", help="Skills action to perform")

        # --- ingest ---
        p_ingest = sub.add_parser("ingest", help="Ingest SKILL.md files from a folder")
        p_ingest.add_argument("skills_folder", help="Path to directory containing skill subdirs")
        p_ingest.add_argument(
            "--dataset", "-d", default="skills", help="Dataset name (default: skills)"
        )
        p_ingest.add_argument("--source-repo", default="", help="Provenance label")
        p_ingest.add_argument(
            "--skip-enrichment",
            action="store_true",
            help="Skip LLM enrichment (parser output only)",
        )

        # --- recommend ---
        p_recommend = sub.add_parser("recommend", help="Find best skills for a task")
        p_recommend.add_argument("task_text", help="Task description in natural language")
        p_recommend.add_argument(
            "--top-k", "-k", type=int, default=5, help="Number of results (default: 5)"
        )
        p_recommend.add_argument(
            "--output-format",
            "-f",
            choices=["pretty", "json"],
            default="pretty",
            help="Output format (default: pretty)",
        )

        # --- list ---
        p_list = sub.add_parser("list", help="List all ingested skills")
        p_list.add_argument(
            "--output-format",
            "-f",
            choices=["pretty", "json"],
            default="pretty",
            help="Output format (default: pretty)",
        )

        # --- observe ---
        p_observe = sub.add_parser("observe", help="Record a skill execution outcome")
        p_observe.add_argument(
            "payload",
            help='JSON with task_text, selected_skill_id, success_score (e.g. \'{"task_text":"...","selected_skill_id":"...","success_score":0.9}\')',
        )

        # --- promote ---
        p_promote = sub.add_parser("promote", help="Promote cached runs to the graph")
        p_promote.add_argument("--session-id", default="", help="Session ID (default: all)")

    def execute(self, args: argparse.Namespace) -> None:
        action = getattr(args, "skills_action", None)
        if not action:
            fmt.error("No skills action specified. Use: ingest, recommend, list, observe, promote")
            raise CliCommandException(
                "Missing skills action. Run 'cognee-cli skills --help' for usage.", error_code=1
            )

        handler = {
            "ingest": self._ingest,
            "recommend": self._recommend,
            "list": self._list,
            "observe": self._observe,
            "promote": self._promote,
        }.get(action)

        if handler is None:
            raise CliCommandException(f"Unknown skills action: {action}", error_code=1)

        handler(args)

    def _ingest(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            fmt.echo(f"Ingesting skills from {args.skills_folder}...")
            asyncio.run(
                skills.ingest(
                    skills_folder=args.skills_folder,
                    dataset_name=args.dataset,
                    source_repo=args.source_repo,
                    skip_enrichment=args.skip_enrichment,
                )
            )
            fmt.success("Skills ingested successfully.")
        except Exception as e:
            raise CliCommandException(f"Failed to ingest skills: {e}", error_code=1) from e

    def _recommend(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            fmt.echo(f"Finding skills for: '{args.task_text}'")
            recs = asyncio.run(skills.get_context(args.task_text, top_k=args.top_k))

            if args.output_format == "json":
                fmt.echo(json.dumps(recs, indent=2, default=str))
                return

            if not recs:
                fmt.warning("No skills found.")
                return

            fmt.echo(f"\nTop {len(recs)} skill(s):")
            fmt.echo("=" * 60)
            for i, rec in enumerate(recs, 1):
                fmt.echo(
                    f"  {i}. {fmt.bold(rec['name'])}  "
                    f"score={rec['score']:.3f}  "
                    f"vector={rec['vector_score']:.3f}  "
                    f"prefers={rec['prefers_score']:.3f}"
                )
                if rec.get("instruction_summary"):
                    fmt.echo(f"     {rec['instruction_summary'][:100]}")
        except Exception as e:
            raise CliCommandException(f"Failed to recommend skills: {e}", error_code=1) from e

    def _list(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            results = asyncio.run(skills.list())

            if args.output_format == "json":
                fmt.echo(json.dumps(results, indent=2, default=str))
                return

            if not results:
                fmt.warning("No skills ingested yet.")
                return

            fmt.echo(f"\n{len(results)} skill(s) ingested:")
            fmt.echo("=" * 60)
            for s in results:
                tags = ", ".join(s.get("tags", [])) if s.get("tags") else ""
                tag_str = f"  [{tags}]" if tags else ""
                fmt.echo(f"  {fmt.bold(s['name'])}{tag_str}")
                if s.get("instruction_summary"):
                    fmt.echo(f"    {s['instruction_summary'][:100]}")
        except Exception as e:
            raise CliCommandException(f"Failed to list skills: {e}", error_code=1) from e

    def _observe(self, args: argparse.Namespace) -> None:
        try:
            payload = json.loads(args.payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CliCommandException(f"Invalid JSON payload: {exc}", error_code=1) from exc

        required = ["task_text", "selected_skill_id", "success_score"]
        missing = [f for f in required if f not in payload]
        if missing:
            raise CliCommandException(
                f"Missing required fields: {', '.join(missing)}", error_code=1
            )

        try:
            from cognee.cognee_skills.client import skills

            result = asyncio.run(skills.observe(payload))
            fmt.success(
                f"Recorded: skill={result.get('selected_skill_id')} "
                f"score={result.get('success_score')}"
            )
        except Exception as e:
            raise CliCommandException(f"Failed to record skill run: {e}", error_code=1) from e

    def _promote(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            sid = args.session_id if args.session_id else None
            result = asyncio.run(skills.promote(session_id=sid))
            fmt.success(
                f"Promoted {result.get('promoted', 0)} run(s), "
                f"updated {result.get('edges_updated', 0)} edge(s)."
            )
        except Exception as e:
            raise CliCommandException(f"Failed to promote skill runs: {e}", error_code=1) from e

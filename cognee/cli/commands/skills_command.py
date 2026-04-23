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
    help_string = "Manage, execute, and self-improve cognee-skills"
    docs_url = DEFAULT_DOCS_URL
    description = """
Manage, execute, and self-improve cognee-skills.

Sub-commands:

  ingest      Parse SKILL.md files from a folder and store in the knowledge graph
  run         Find the best skill for a task and execute it
  execute     Execute a specific skill by ID
  list        List all ingested skills
  recommend   Find the best skills for a task using semantic search + learned preferences
  inspect     Analyze why a skill keeps failing
  preview     Preview a proposed fix for a failing skill
  amendify    Apply a proposed amendment to a skill
  rollback    Revert an applied amendment
  evaluate    Compare before/after scores for an amendment
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

        # --- run ---
        p_run = sub.add_parser("run", help="Find the best skill for a task and execute it")
        p_run.add_argument("task_text", help="Task description in natural language")
        p_run.add_argument("--context", "-c", default="", help="Additional context")
        p_run.add_argument(
            "--no-evaluate", action="store_true", help="Skip output quality evaluation"
        )
        p_run.add_argument("--no-amendify", action="store_true", help="Skip automatic self-repair")
        p_run.add_argument(
            "--output-format",
            "-f",
            choices=["pretty", "json"],
            default="pretty",
            help="Output format (default: pretty)",
        )

        # --- execute ---
        p_execute = sub.add_parser("execute", help="Execute a specific skill by ID")
        p_execute.add_argument("skill_id", help="Skill ID to execute")
        p_execute.add_argument("task_text", help="Task description in natural language")
        p_execute.add_argument("--context", "-c", default="", help="Additional context")
        p_execute.add_argument(
            "--no-evaluate", action="store_true", help="Skip output quality evaluation"
        )
        p_execute.add_argument(
            "--output-format",
            "-f",
            choices=["pretty", "json"],
            default="pretty",
            help="Output format (default: pretty)",
        )

        # --- inspect ---
        p_inspect = sub.add_parser("inspect", help="Analyze why a skill keeps failing")
        p_inspect.add_argument("skill_id", help="Skill ID to inspect")
        p_inspect.add_argument(
            "--min-runs", type=int, default=1, help="Minimum failed runs required (default: 1)"
        )
        p_inspect.add_argument(
            "--score-threshold",
            type=float,
            default=0.5,
            help="Runs below this score count as failures (default: 0.5)",
        )
        p_inspect.add_argument(
            "--output-format",
            "-f",
            choices=["pretty", "json"],
            default="pretty",
            help="Output format (default: pretty)",
        )

        # --- preview ---
        p_preview = sub.add_parser("preview", help="Preview a proposed fix for a failing skill")
        p_preview.add_argument("skill_id", help="Skill ID to generate a fix for")
        p_preview.add_argument(
            "--min-runs", type=int, default=1, help="Minimum failed runs required (default: 1)"
        )
        p_preview.add_argument(
            "--score-threshold",
            type=float,
            default=0.5,
            help="Runs below this score count as failures (default: 0.5)",
        )
        p_preview.add_argument(
            "--output-format",
            "-f",
            choices=["pretty", "json"],
            default="pretty",
            help="Output format (default: pretty)",
        )

        # --- amendify ---
        p_amendify = sub.add_parser("amendify", help="Apply a proposed amendment to a skill")
        p_amendify.add_argument("amendment_id", help="Amendment ID to apply")
        p_amendify.add_argument(
            "--write-to-disk", action="store_true", help="Also update SKILL.md on disk"
        )

        # --- rollback ---
        p_rollback = sub.add_parser("rollback", help="Revert an applied amendment")
        p_rollback.add_argument("amendment_id", help="Amendment ID to rollback")
        p_rollback.add_argument(
            "--write-to-disk", action="store_true", help="Also restore SKILL.md on disk"
        )

        # --- evaluate ---
        p_evaluate = sub.add_parser("evaluate", help="Compare before/after scores for an amendment")
        p_evaluate.add_argument("amendment_id", help="Amendment ID to evaluate")

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
            fmt.error(
                "No skills action specified. Use: ingest, run, execute, list, recommend, "
                "inspect, preview, amendify, rollback, evaluate, observe, promote"
            )
            raise CliCommandException(
                "Missing skills action. Run 'cognee-cli skills --help' for usage.", error_code=1
            )

        handler = {
            "ingest": self._ingest,
            "run": self._run,
            "execute": self._execute,
            "list": self._list,
            "recommend": self._recommend,
            "inspect": self._inspect,
            "preview": self._preview,
            "amendify": self._amendify,
            "rollback": self._rollback,
            "evaluate": self._evaluate,
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

    def _run(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            is_json = args.output_format == "json"
            if not is_json:
                fmt.echo(f"Running: '{args.task_text}'")

            result = asyncio.run(
                skills.run(
                    task_text=args.task_text,
                    context=args.context or None,
                    auto_evaluate=not args.no_evaluate,
                    auto_amendify=not args.no_amendify,
                )
            )

            if is_json:
                fmt.echo(json.dumps(result, indent=2, default=str))
                return

            fmt.echo(f"\n  Skill:   {result.get('skill_id', '?')}")
            fmt.echo(f"  Quality: {result.get('quality_score', 'N/A')}")
            if result.get("quality_reason"):
                fmt.echo(f"  Reason:  {result['quality_reason']}")
            fmt.echo(f"\n{result.get('output', '')}")

            if result.get("amended"):
                fmt.note(f"  Self-repaired: {result['amended'].get('amendment_id', '')}")
        except Exception as e:
            raise CliCommandException(f"Failed to run skill: {e}", error_code=1) from e

    def _execute(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            is_json = args.output_format == "json"
            if not is_json:
                fmt.echo(f"Executing skill '{args.skill_id}': '{args.task_text}'")

            result = asyncio.run(
                skills.execute(
                    skill_id=args.skill_id,
                    task_text=args.task_text,
                    context=args.context or None,
                    auto_evaluate=not args.no_evaluate,
                )
            )

            if is_json:
                fmt.echo(json.dumps(result, indent=2, default=str))
                return

            success = result.get("success", False)
            if success:
                fmt.echo(f"\n  Quality: {result.get('quality_score', 'N/A')}")
                if result.get("quality_reason"):
                    fmt.echo(f"  Reason:  {result['quality_reason']}")
                fmt.echo(f"\n{result.get('output', '')}")
            else:
                fmt.error(f"  Error: {result.get('error', 'unknown')}")
        except Exception as e:
            raise CliCommandException(f"Failed to execute skill: {e}", error_code=1) from e

    def _inspect(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            is_json = args.output_format == "json"
            if not is_json:
                fmt.echo(f"Inspecting skill '{args.skill_id}'...")

            result = asyncio.run(
                skills.inspect(
                    skill_id=args.skill_id,
                    min_runs=args.min_runs,
                    score_threshold=args.score_threshold,
                )
            )

            if result is None:
                if is_json:
                    fmt.echo("null")
                else:
                    fmt.warning("No failures detected (not enough low-scoring runs).")
                return

            if is_json:
                fmt.echo(json.dumps(result, indent=2, default=str))
                return

            fmt.echo(f"\n  Failure category: {result['failure_category']}")
            fmt.echo(f"  Severity:         {result['severity']}")
            fmt.echo(f"  Avg score:        {result['avg_success_score']:.2f}")
            fmt.echo(f"  Runs analyzed:    {result['analyzed_run_count']}")
            fmt.echo(f"  Root cause:       {result['root_cause']}")
            fmt.echo(f"  Suggestion:       {result['improvement_hypothesis']}")
        except Exception as e:
            raise CliCommandException(f"Failed to inspect skill: {e}", error_code=1) from e

    def _preview(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            is_json = args.output_format == "json"
            if not is_json:
                fmt.echo(f"Generating fix for skill '{args.skill_id}'...")

            result = asyncio.run(
                skills.preview_amendify(
                    skill_id=args.skill_id,
                    min_runs=args.min_runs,
                    score_threshold=args.score_threshold,
                )
            )

            if result is None:
                if is_json:
                    fmt.echo("null")
                else:
                    fmt.warning("No amendment proposed (inspection found no issues).")
                return

            if is_json:
                fmt.echo(json.dumps(result, indent=2, default=str))
                return

            fmt.echo(f"\n  Amendment ID: {result['amendment_id']}")
            fmt.echo(f"  Confidence:   {result['amendment_confidence']}")
            fmt.echo(f"  Explanation:  {result['change_explanation']}")

            fmt.echo(f"\n--- ORIGINAL INSTRUCTIONS ---")
            fmt.echo(result.get("original_instructions", "N/A"))
            fmt.echo(f"\n--- PROPOSED INSTRUCTIONS ---")
            fmt.echo(result.get("amended_instructions", "N/A"))

            fmt.note(f"\nTo apply: cognee-cli skills amendify {result['amendment_id']}")
        except Exception as e:
            raise CliCommandException(f"Failed to preview amendment: {e}", error_code=1) from e

    def _amendify(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            fmt.echo(f"Applying amendment '{args.amendment_id}'...")
            result = asyncio.run(
                skills.amendify(
                    amendment_id=args.amendment_id,
                    write_to_disk=args.write_to_disk,
                )
            )

            if result.get("success"):
                fmt.success(
                    f"Applied amendment to skill '{result.get('skill_name', '?')}' "
                    f"(status: {result.get('status')})"
                )
            else:
                fmt.error(f"Failed: {result.get('error', 'unknown')}")
        except Exception as e:
            raise CliCommandException(f"Failed to apply amendment: {e}", error_code=1) from e

    def _rollback(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            fmt.echo(f"Rolling back amendment '{args.amendment_id}'...")
            result = asyncio.run(
                skills.rollback_amendify(
                    amendment_id=args.amendment_id,
                    write_to_disk=args.write_to_disk,
                )
            )

            if result:
                fmt.success("Amendment rolled back. Original instructions restored.")
            else:
                fmt.error("Rollback failed (amendment not found or not in 'applied' state).")
        except Exception as e:
            raise CliCommandException(f"Failed to rollback amendment: {e}", error_code=1) from e

    def _evaluate(self, args: argparse.Namespace) -> None:
        try:
            from cognee.cognee_skills.client import skills

            fmt.echo(f"Evaluating amendment '{args.amendment_id}'...")
            result = asyncio.run(skills.evaluate_amendify(amendment_id=args.amendment_id))

            if result.get("error"):
                fmt.error(result["error"])
                return

            fmt.echo(f"\n  Pre-amendment avg:  {result['pre_avg']:.2f}")
            fmt.echo(f"  Post-amendment avg: {result['post_avg']:.2f}")
            fmt.echo(f"  Improvement:        {result['improvement']:+.2f}")
            fmt.echo(f"  Post-amendment runs: {result['run_count']}")
            rec = result["recommendation"]
            if rec == "keep":
                fmt.success(f"  Recommendation: {rec}")
            else:
                fmt.warning(f"  Recommendation: {rec}")
        except Exception as e:
            raise CliCommandException(f"Failed to evaluate amendment: {e}", error_code=1) from e

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

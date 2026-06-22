#!/usr/bin/env python3
"""
Generate human-friendly release notes by analyzing git diff with LLM.

This script compares changes between two branches (typically main and dev)
and uses an LLM to generate readable, user-focused release notes.

Uses litellm + instructor directly to avoid cognee's dotenv.load_dotenv(override=True)
which can overwrite CI environment variables with .env file placeholders.
"""

import argparse
import asyncio
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def run_git_command(command: list[str]) -> str:
    """Execute a git command and return output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}", file=sys.stderr)
        print(f"Command output: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def get_latest_release_tag() -> str | None:
    """Get the latest release tag."""
    try:
        # Get the latest tag that matches version pattern (vX.Y.Z)
        command = ["git", "tag", "--sort=-version:refname", "--list", "v*"]
        tags = run_git_command(command)
        if tags:
            return tags.split("\n")[0]
        return None
    except Exception:
        return None


def get_git_diff(base_ref: str, target_ref: str, include_stats: bool = True) -> str:
    """Get git diff between two refs (branches, tags, commits)."""
    # Get diff with file stats
    diff_command = ["git", "diff", f"{base_ref}...{target_ref}"]
    if include_stats:
        diff_command.append("--stat")

    diff_output = run_git_command(diff_command)

    # Get detailed diff (limited to avoid token overflow)
    detailed_diff_command = [
        "git",
        "diff",
        f"{base_ref}...{target_ref}",
        "--unified=3",  # 3 lines of context
        "--no-color",
    ]
    detailed_diff = run_git_command(detailed_diff_command)

    return f"STATISTICS:\n{diff_output}\n\nDETAILED CHANGES:\n{detailed_diff}"


def get_commit_history(base_ref: str, target_ref: str) -> str:
    """Get commit messages between two refs (branches, tags, commits)."""
    command = [
        "git",
        "log",
        f"{base_ref}..{target_ref}",
        "--pretty=format:- %s (%h) by %an",
        "--no-merges",
    ]
    return run_git_command(command)


def get_pr_list(base_ref: str, target_ref: str) -> str:
    """Get list of PR numbers from commit messages."""
    command = [
        "git",
        "log",
        f"{base_ref}..{target_ref}",
        "--pretty=format:%s",
        "--no-merges",
    ]
    commits = run_git_command(command)

    # Extract PR numbers (matches #1234 pattern)
    pr_numbers = set(re.findall(r"#(\d+)", commits))
    return ", ".join(f"#{pr}" for pr in sorted(pr_numbers))


def _parse_dependencies(pyproject_text: str) -> dict[str, str]:
    """Parse direct project dependencies into a {package: specifier} map.

    Uses tomllib when available (Python 3.11+), falling back to a line-based
    regex parse so the script also works on Python 3.10.
    """
    deps: dict[str, str] = {}

    def _split(req: str) -> tuple[str, str]:
        # Strip environment markers (after ';') and extras ('pkg[extra]').
        req = req.split(";")[0].strip()
        match = re.match(r"^([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(.*)$", req)
        if not match:
            return req, ""
        return match.group(1).lower(), match.group(2).strip()

    try:
        import tomllib  # type: ignore[import-not-found]

        data = tomllib.loads(pyproject_text)
        for req in data.get("project", {}).get("dependencies", []):
            name, spec = _split(req)
            if name:
                deps[name] = spec
        return deps
    except Exception:
        pass

    # Fallback: extract the [project].dependencies = [ ... ] array textually.
    match = re.search(r"^dependencies\s*=\s*\[(.*?)\]", pyproject_text, re.DOTALL | re.MULTILINE)
    if not match:
        return deps
    for raw in re.findall(r"""["']([^"']+)["']""", match.group(1)):
        name, spec = _split(raw)
        if name:
            deps[name] = spec
    return deps


def get_dependency_changes(base_ref: str, target_ref: str) -> dict[str, list[str]]:
    """Compare direct dependencies in pyproject.toml between two refs."""
    changes: dict[str, list[str]] = {"added": [], "removed": [], "changed": []}
    try:
        base_text = run_git_command(["git", "show", f"{base_ref}:pyproject.toml"])
        target_text = run_git_command(["git", "show", f"{target_ref}:pyproject.toml"])
    except SystemExit:
        return changes
    except Exception:
        return changes

    base_deps = _parse_dependencies(base_text)
    target_deps = _parse_dependencies(target_text)

    for name, spec in sorted(target_deps.items()):
        if name not in base_deps:
            changes["added"].append(f"{name} {spec}".strip())
        elif base_deps[name] != spec:
            changes["changed"].append(f"{name}: {base_deps[name] or '*'} → {spec or '*'}")
    for name, spec in sorted(base_deps.items()):
        if name not in target_deps:
            changes["removed"].append(f"{name} {spec}".strip())
    return changes


def get_compatibility_info(target_ref: str) -> dict[str, str]:
    """Read compatibility-relevant metadata from pyproject.toml at the target ref."""
    info: dict[str, str] = {}
    try:
        text = run_git_command(["git", "show", f"{target_ref}:pyproject.toml"])
    except Exception:
        try:
            text = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        except Exception:
            return info

    py_match = re.search(r'^requires-python\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if py_match:
        info["python"] = py_match.group(1)

    # Surface a few core runtime dependencies users commonly pin against.
    deps = _parse_dependencies(text)
    for pkg in ("pydantic", "litellm", "fastapi", "sqlalchemy", "lancedb", "ladybug"):
        if pkg in deps:
            info[pkg] = deps[pkg] or "*"
    return info


async def generate_release_notes_with_llm(
    diff: str,
    commits: str,
    base_ref: str,
    target_ref: str,
    version: str | None = None,
) -> Any:
    """Use LLM to generate human-friendly release notes.

    Uses litellm + instructor directly instead of cognee's LLM client
    to avoid cognee's dotenv.load_dotenv(override=True) overwriting
    CI environment variables.
    """
    try:
        import instructor
        import litellm
        from pydantic import BaseModel, Field
    except ImportError as e:
        print(f"Error: Required dependencies not available: {e}", file=sys.stderr)
        print(
            "Please ensure litellm, instructor, and pydantic are installed.",
            file=sys.stderr,
        )
        return None

    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")

    if not api_key:
        print("Warning: LLM_API_KEY not set, skipping LLM generation.", file=sys.stderr)
        return None

    class ReleaseNotes(BaseModel):
        """Structured release notes."""

        title: str = Field(
            description=(
                "A short, descriptive release title naming the single most important theme "
                "of this release, formatted as 'v{version} — {Theme}' "
                "(e.g. 'v0.1.9 — Memory Graph Improvements'). "
                "If no single theme dominates, summarize the largest change area. "
                "Keep it under 60 characters and never leave it empty."
            )
        )
        summary: str = Field(description="A brief, engaging summary of the release (2-3 sentences)")
        highlights: list[str] = Field(description="3-5 key highlights that users care about most")
        features: list[str] = Field(
            description=(
                "New features added in this release. Each entry must be understandable to a "
                "first-time reader: name the feature, say in plain words what it does, and why "
                "it matters. Do not use undefined jargon."
            )
        )
        improvements: list[str] = Field(
            description="Enhancements and improvements to existing functionality"
        )
        performance: list[str] = Field(
            description=(
                "Performance changes: speedups, reduced memory/latency, throughput gains. "
                "State the practical impact (e.g. 'faster ingestion of large files')."
            )
        )
        security: list[str] = Field(
            description=(
                "Security-relevant changes: hardening, auth fixes, vulnerability patches. "
                "Do NOT invent CVE identifiers — only mention a CVE if it appears in the diff "
                "or commit messages."
            )
        )
        bug_fixes: list[str] = Field(description="Bug fixes and issue resolutions")
        breaking_changes: list[str] = Field(description="Breaking changes that require user action")
        technical_changes: list[str] = Field(
            description="Technical changes, refactoring, and internal improvements"
        )

    # Truncate diff if too large (keep first 15000 chars)
    max_diff_size = 15000
    if len(diff) > max_diff_size:
        diff = diff[:max_diff_size] + f"\n... (truncated, showing first {max_diff_size} characters)"

    system_prompt = """You are a technical writer creating release notes for Cognee, an AI memory platform.

Analyze the git diff and commit history to create clear, user-friendly release notes.

WRITING FOR A FIRST-TIME READER (most important rule):
- Assume the reader has never seen this feature before and does not know Cognee's internal vocabulary.
- Every entry must answer, in plain words: WHAT it is, WHAT it does, and WHY it matters.
- Do NOT use undefined jargon. If a term like "summary layer", "dataset", "graph completion",
  or "semantic bucket" is unavoidable, explain it in the same sentence the first time it appears.
- Bad: "An optional summary layer that builds dataset-level orientation through semantic buckets."
  Good: "A new optional index that groups a dataset (a collection of documents you've added) into
  topic clusters and a short overview, so search has broader context and returns better answers."
- Prefer concrete outcomes ("search returns more relevant results") over abstract descriptions.

Other guidelines:
- Choose ONE dominant theme for the title; if none dominates, name the largest change area.
- Focus on user-facing changes and their benefits.
- Group related changes together and categorize each change correctly
  (feature / improvement / performance / security / bug fix / breaking change / technical).
- Highlight breaking changes prominently.
- Be concise but informative; emphasize the "why" and "what it means", not just "what changed".
- Mention specific file names only if critical to understanding.
- Never invent CVE numbers, deprecation dates, or version-support promises that are not in the input.
"""

    text_input = f"""Generate release notes for Cognee version {version or "TBD"}.

Base reference: {base_ref}
Target reference: {target_ref}

COMMIT HISTORY:
{commits[:5000]}

GIT DIFF:
{diff}

Create engaging release notes that help users understand what's new and improved.
"""

    try:
        client = instructor.from_litellm(litellm.acompletion)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input},
            ],
            response_model=ReleaseNotes,
            api_key=api_key,
            max_retries=2,
        )
        return response
    except Exception as e:
        print(f"Warning: LLM generation failed: {e}", file=sys.stderr)
        return None


def generate_fallback_notes(
    commits: str,
    version: str,
) -> Any:
    """Generate basic release notes from commit messages when LLM is unavailable."""
    from dataclasses import dataclass, field

    @dataclass
    class FallbackNotes:
        title: str = ""
        summary: str = ""
        highlights: list[str] = field(default_factory=list)
        features: list[str] = field(default_factory=list)
        improvements: list[str] = field(default_factory=list)
        performance: list[str] = field(default_factory=list)
        security: list[str] = field(default_factory=list)
        bug_fixes: list[str] = field(default_factory=list)
        breaking_changes: list[str] = field(default_factory=list)
        technical_changes: list[str] = field(default_factory=list)

    notes = FallbackNotes()

    for line in commits.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("- "):
            continue
        msg = line[2:]  # strip "- " prefix
        lower = msg.lower()

        if lower.startswith("feat"):
            notes.features.append(msg)
        elif lower.startswith("fix"):
            notes.bug_fixes.append(msg)
        elif lower.startswith("perf"):
            notes.performance.append(msg)
        elif lower.startswith(("security", "sec(")):
            notes.security.append(msg)
        elif lower.startswith("breaking"):
            notes.breaking_changes.append(msg)
        elif any(lower.startswith(p) for p in ("refactor", "chore", "ci", "build", "test")):
            notes.technical_changes.append(msg)
        else:
            notes.improvements.append(msg)

    notes.title = f"v{version} — {len(notes.features)} features, {len(notes.bug_fixes)} fixes"
    notes.summary = f"Cognee v{version} includes {len(notes.features)} new features, {len(notes.bug_fixes)} bug fixes, and {len(notes.improvements)} improvements."
    notes.highlights = (notes.features + notes.improvements)[:5]

    return notes


def get_release_title(notes: Any, version: str) -> str:
    """Resolve the release title, guaranteeing a non-empty, version-prefixed value."""
    # Collapse any whitespace/newlines so the title is safe as a single-line
    # GitHub Actions output (RELEASE_TITLE=...).
    title = " ".join((getattr(notes, "title", "") or "").split())
    if not title:
        return f"v{version}"
    # Ensure the version prefix is present so releases sort/scan consistently.
    if not re.match(rf"^v?{re.escape(version)}\b", title):
        return f"v{version} — {title}"
    return title if title.startswith("v") else f"v{title}"


def format_release_notes(
    notes: Any,
    version: str,
    base_ref: str,
    target_ref: str,
    pr_list: str,
    dep_changes: dict[str, list[str]] | None = None,
    compat_info: dict[str, str] | None = None,
) -> str:
    """Format structured release notes into markdown."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    md = f"# {get_release_title(notes, version)}\n\n"
    md += f"**Release Date:** {date_str}\n"
    md += f"**Changes:** {base_ref} → {target_ref}\n\n"

    if pr_list:
        md += f"**Pull Requests:** {pr_list}\n\n"

    md += "---\n\n"

    # Summary
    md += f"## Summary\n\n{notes.summary}\n\n"

    # Highlights
    if notes.highlights:
        md += "## Highlights\n\n"
        for highlight in notes.highlights:
            md += f"- {highlight}\n"
        md += "\n"

    def _section(heading: str, items: list[str]) -> str:
        if not items:
            return ""
        block = f"## {heading}\n\n"
        for item in items:
            block += f"- {item}\n"
        return block + "\n"

    # Ordered by impact.
    md += _section("Breaking Changes", notes.breaking_changes)
    md += _section("New Features", notes.features)
    md += _section("Improvements", notes.improvements)
    md += _section("Performance", getattr(notes, "performance", []))
    md += _section("Security", getattr(notes, "security", []))
    md += _section("Bug Fixes", notes.bug_fixes)
    md += _section("Technical Changes", notes.technical_changes)

    # Dependency updates (deterministic, parsed from pyproject.toml diff).
    if dep_changes and any(dep_changes.values()):
        md += "## Dependency Updates\n\n"
        if dep_changes.get("added"):
            md += "**Added:**\n"
            for dep in dep_changes["added"]:
                md += f"- {dep}\n"
            md += "\n"
        if dep_changes.get("changed"):
            md += "**Updated:**\n"
            for dep in dep_changes["changed"]:
                md += f"- {dep}\n"
            md += "\n"
        if dep_changes.get("removed"):
            md += "**Removed:**\n"
            for dep in dep_changes["removed"]:
                md += f"- {dep}\n"
            md += "\n"

    # Compatibility matrix (deterministic, read from pyproject.toml).
    if compat_info:
        md += "## Compatibility\n\n"
        md += "| Component | Supported / Required |\n"
        md += "| --- | --- |\n"
        if compat_info.get("python"):
            md += f"| Python | `{compat_info['python']}` |\n"
        for pkg, spec in compat_info.items():
            if pkg == "python":
                continue
            md += f"| {pkg} | `{spec}` |\n"
        md += "\n"

    # Footer
    md += "---\n\n"
    md += f"*— The Cognee Team · {date_str}*\n"

    return md


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate AI-powered release notes from git diff")
    parser.add_argument(
        "--base",
        help="Base ref to compare against (tag, branch, or commit). If not provided, uses latest release tag",
    )
    parser.add_argument(
        "--target",
        help="Target ref with new changes (tag, branch, or commit). Defaults to current branch",
    )
    parser.add_argument(
        "--version",
        help="Release version number (if not provided, will be extracted from pyproject.toml)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Format output for GitHub Actions (sets RELEASE_NOTES environment variable)",
    )

    args = parser.parse_args()

    # Get version from pyproject.toml if not provided
    version = args.version
    if not version:
        try:
            # Read version directly from pyproject.toml
            pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
            content = pyproject_path.read_text()
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                version = match.group(1)
            else:
                version = "unknown"
        except Exception as e:
            print(f"Warning: Could not extract version: {e}", file=sys.stderr)
            version = "unknown"

    # Determine base ref (what to compare against)
    base_ref = args.base
    if not base_ref:
        # Use latest release tag as base
        latest_tag = get_latest_release_tag()
        if latest_tag:
            base_ref = latest_tag
            print(f"Using latest release tag as base: {base_ref}", file=sys.stderr)
        else:
            print("Warning: No release tags found, comparing against main branch", file=sys.stderr)
            base_ref = "main"

    # Determine target ref (what we're releasing)
    target_ref = args.target
    if not target_ref:
        # Use current branch
        target_ref = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        print(f"Using current branch as target: {target_ref}", file=sys.stderr)

    print(f"Generating release notes for version {version}...", file=sys.stderr)
    print(f"Comparing {base_ref}...{target_ref}", file=sys.stderr)

    # Get git data
    diff = get_git_diff(base_ref, target_ref)
    commits = get_commit_history(base_ref, target_ref)
    pr_list = get_pr_list(base_ref, target_ref)
    dep_changes = get_dependency_changes(base_ref, target_ref)
    compat_info = get_compatibility_info(target_ref)

    # Generate release notes with LLM, fall back to commit-based notes
    notes = await generate_release_notes_with_llm(diff, commits, base_ref, target_ref, version)

    if notes is None:
        print("Falling back to commit-based release notes.", file=sys.stderr)
        notes = generate_fallback_notes(commits, version)

    # Format as markdown
    title = get_release_title(notes, version)
    markdown = format_release_notes(
        notes, version, base_ref, target_ref, pr_list, dep_changes, compat_info
    )

    # Output results
    if args.github_output:
        # GitHub Actions multi-line output
        delimiter = "EOF"
        output_var = f"RELEASE_NOTES<<{delimiter}\n{markdown}\n{delimiter}\n"
        output_var += f"RELEASE_TITLE={title}\n"

        # Write to GITHUB_OUTPUT file
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(output_var)
            print("Release notes written to GITHUB_OUTPUT", file=sys.stderr)
        else:
            print(output_var)
    elif args.output:
        with open(args.output, "w") as f:
            f.write(markdown)
        print(f"Release notes written to {args.output}", file=sys.stderr)
    else:
        print(markdown)

    print("\nRelease notes generated successfully!", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())

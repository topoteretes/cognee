#!/usr/bin/env python3
"""
Generate human-friendly release notes by analyzing git diff with LLM.

This script compares changes between two branches (typically main and dev)
and uses an LLM to generate readable, user-focused release notes.
"""

import argparse
import asyncio
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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


def get_latest_release_tag() -> str:
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
    import re

    pr_numbers = set(re.findall(r"#(\d+)", commits))
    return ", ".join(f"#{pr}" for pr in sorted(pr_numbers))


async def generate_release_notes_with_llm(
    diff: str,
    commits: str,
    base_ref: str,
    target_ref: str,
    version: str = None,
) -> str:
    """Use LLM to generate human-friendly release notes."""
    try:
        # Import here to avoid startup delays
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )
        from pydantic import BaseModel, Field
    except ImportError as e:
        print(f"Error: Required dependencies not available: {e}", file=sys.stderr)
        print("Please ensure cognee is installed with: pip install -e .", file=sys.stderr)
        sys.exit(1)

    class ReleaseNotes(BaseModel):
        """Structured release notes."""

        summary: str = Field(description="A brief, engaging summary of the release (2-3 sentences)")
        highlights: list[str] = Field(description="3-5 key highlights that users care about most")
        features: list[str] = Field(description="New features added in this release")
        improvements: list[str] = Field(
            description="Enhancements and improvements to existing functionality"
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

    system_prompt = f"""You are a technical writer creating release notes for Cognee, an AI memory platform.

Analyze the git diff and commit history to create clear, user-friendly release notes.

Guidelines:
- Focus on user-facing changes and their benefits
- Group related changes together
- Use clear, non-technical language where possible
- Highlight breaking changes prominently
- Organize by impact: features > improvements > bug fixes > technical changes
- Be concise but informative
- Emphasize the "why" and "what it means" for users, not just "what changed"
- Mention specific file names only if critical to understanding
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
        llm_client = get_llm_client()
        response = await llm_client.acreate_structured_output(
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=ReleaseNotes,
        )

        return response
    except Exception as e:
        print(f"Error generating release notes with LLM: {e}", file=sys.stderr)
        sys.exit(1)


def format_release_notes(
    notes: any, version: str, base_ref: str, target_ref: str, pr_list: str
) -> str:
    """Format structured release notes into markdown."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    md = f"# Release Notes - v{version}\n\n"
    md += f"**Release Date:** {date_str}\n"
    md += f"**Changes:** {base_ref} → {target_ref}\n\n"

    if pr_list:
        md += f"**Pull Requests:** {pr_list}\n\n"

    md += "---\n\n"

    # Summary
    md += f"## 🎉 Summary\n\n{notes.summary}\n\n"

    # Highlights
    if notes.highlights:
        md += "## ⭐ Highlights\n\n"
        for highlight in notes.highlights:
            md += f"- {highlight}\n"
        md += "\n"

    # Breaking Changes (if any)
    if notes.breaking_changes:
        md += "## ⚠️ Breaking Changes\n\n"
        for change in notes.breaking_changes:
            md += f"- {change}\n"
        md += "\n"

    # Features
    if notes.features:
        md += "## ✨ New Features\n\n"
        for feature in notes.features:
            md += f"- {feature}\n"
        md += "\n"

    # Improvements
    if notes.improvements:
        md += "## 🚀 Improvements\n\n"
        for improvement in notes.improvements:
            md += f"- {improvement}\n"
        md += "\n"

    # Bug Fixes
    if notes.bug_fixes:
        md += "## 🐛 Bug Fixes\n\n"
        for fix in notes.bug_fixes:
            md += f"- {fix}\n"
        md += "\n"

    # Technical Changes
    if notes.technical_changes:
        md += "## 🔧 Technical Changes\n\n"
        for change in notes.technical_changes:
            md += f"- {change}\n"
        md += "\n"

    # Footer
    md += "---\n\n"
    md += f"*Generated by Cognee AI Release Notes Generator on {date_str}*\n"

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
            import re

            pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
            with open(pyproject_path, "r") as f:
                content = f.read()
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

    # Generate release notes with LLM
    notes = await generate_release_notes_with_llm(diff, commits, base_ref, target_ref, version)

    # Format as markdown
    markdown = format_release_notes(notes, version, base_ref, target_ref, pr_list)

    # Output results
    if args.github_output:
        # GitHub Actions multi-line output
        delimiter = "EOF"
        output_var = f"RELEASE_NOTES<<{delimiter}\n{markdown}\n{delimiter}\n"

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

    print("\n✅ Release notes generated successfully!", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())

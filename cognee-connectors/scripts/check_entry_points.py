#!/usr/bin/env python3
"""Resolve the entry points an installed connector distribution declares.

Hosts find a package through three entry-point groups (docs/ARCHITECTURE.md, rule 6):

  - ``cognee.connectors``: ``<provider> = cognee_connector_<p>:CONNECTOR`` (the contract object)
  - ``cognee.sources``: ``<source> = cognee_connector_<p>:<source>_source`` (one per DLT source)
  - ``cognee.integrations``: ``<provider> = cognee_connector_<p>.cognee_host:register``

The first two are loaded with ``EntryPoint.load()``. ``cognee.integrations`` registers the
cognee_host adapter, which needs cognee: it is loaded only when cognee is importable and
skipped with a message otherwise (the clean-install venv has no cognee by design).

Usage:
    python scripts/check_entry_points.py cognee-connector-notion
    python scripts/check_entry_points.py cognee-connector-notion --require-group cognee.connectors

Exit status: 0 when every declared entry point resolved (and every required group has one);
1 otherwise; 2 when the distribution is not installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from importlib.metadata import PackageNotFoundError, distribution

# A failed load is reported through logger.exception, which prints its traceback too
# (configured to plain stdout lines in main()).
logger = logging.getLogger(__name__)

GROUPS = ("cognee.connectors", "cognee.sources", "cognee.integrations")
HOST_GROUP = "cognee.integrations"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Load the cognee entry points of an installed distribution."
    )
    parser.add_argument("distribution", help="the distribution name, e.g. cognee-connector-notion")
    parser.add_argument(
        "--require-group",
        action="append",
        default=[],
        metavar="GROUP",
        help="fail when this group declares no entry point (repeatable)",
    )
    args = parser.parse_args(argv)

    try:
        dist = distribution(args.distribution)
    except PackageNotFoundError:
        print(f"error: distribution {args.distribution!r} is not installed in {sys.executable}")
        return 2

    cognee_available = importlib.util.find_spec("cognee") is not None
    failures = 0
    declared = 0
    for group in GROUPS:
        entry_points = [ep for ep in dist.entry_points if ep.group == group]
        if not entry_points:
            print(f"{group}: none declared")
            if group in args.require_group:
                print(f"FAIL: {args.distribution} must declare an entry point in {group}")
                failures += 1
            continue
        declared += len(entry_points)
        if group == HOST_GROUP and not cognee_available:
            print(
                f"{group}: skipped, cognee is not installed "
                f"({', '.join(f'{ep.name} = {ep.value}' for ep in entry_points)})"
            )
            continue
        for entry_point in entry_points:
            try:
                target = entry_point.load()
            except BaseException:  # every failure is reported, not hidden
                logger.exception("FAIL: %s: %s = %s", group, entry_point.name, entry_point.value)
                failures += 1
                continue
            print(f"{group}: {entry_point.name} = {entry_point.value} -> {target!r}")

    if failures:
        print(f"\n{failures} entry point problem(s) in {args.distribution}.")
        return 1
    if declared == 0:
        print(f"\n{args.distribution} declares no entry point in {', '.join(GROUPS)}.")
    else:
        print(f"\nEvery entry point of {args.distribution} resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

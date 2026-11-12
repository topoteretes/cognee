"""Install what the checkout's cognee needs and the connectors lock does not have.

The workspace locks cognee from PyPI, so a CI job that swaps in the checkout's cognee
with ``--no-deps`` runs on the published release's dependency set. A change under
``cognee/`` that adds or raises a runtime dependency would then fail every connector
suite with an ImportError no connector caused, and the connectors lock cannot fix it:
it cannot resolve an unreleased cognee.

So this adds, at the version the repository root's ``uv.lock`` pins, each dependency
that is missing and each direct cognee requirement the environment does not satisfy.
Everything else stays as the connectors lock installed it. Run it from
``cognee-connectors/``, after installing cognee from the checkout.
"""

import argparse
import logging
import subprocess
from importlib.metadata import PackageNotFoundError, requires, version

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

logger = logging.getLogger(__name__)


def _installed_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _locked_requirements() -> dict[str, str]:
    """The root lock's pins for this interpreter, as ``{normalized name: requirement}``."""
    exported = subprocess.run(
        [
            "uv",
            "export",
            "--project",
            "..",
            "--frozen",
            "--no-hashes",
            "--no-dev",
            "--no-emit-project",
            "--no-header",
            "--no-annotate",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pins: dict[str, str] = {}
    for line in exported.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            # The marker is settled for this interpreter and has no place on a command line.
            requirement.marker = None
            pins[canonicalize_name(requirement.name)] = str(requirement)
    return pins


def _unsatisfied_direct_requirements() -> set[str]:
    """Names of cognee's own runtime requirements the installed versions do not meet."""
    unsatisfied: set[str] = set()
    for line in requires("cognee") or []:
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        installed = _installed_version(requirement.name)
        if installed is not None and not requirement.specifier.contains(
            installed, prereleases=True
        ):
            unsatisfied.add(canonicalize_name(requirement.name))
    return unsatisfied


def main() -> int:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="only list what would be installed")
    args = parser.parse_args()

    pins = _locked_requirements()
    wanted = {name for name in pins if _installed_version(name) is None}
    wanted |= _unsatisfied_direct_requirements() & pins.keys()
    requirements = sorted(pins[name] for name in wanted)

    if not requirements:
        logger.info("The environment already has everything the checkout's cognee needs.")
        return 0
    logger.info("Installing from the root lock: %s", ", ".join(requirements))
    if not args.dry_run:
        # --no-deps: the root lock's export is already the full closure.
        subprocess.run(["uv", "pip", "install", "--no-deps", *requirements], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

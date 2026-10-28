#!/usr/bin/env python3
"""Scaffold a new connector package from templates/connector/.

Creates ``connectors/<slug>/`` with a pyproject.toml, README, LICENSE (a copy of the
repository's), the ``cognee_connector_<slug>`` module (``CONNECTOR`` from a tier-1
descriptor, settings, a DLT source skeleton, ``cognee_host/`` with ``register()`` and
``routers()``) and a first contract test, appends a ``planned`` entry to inventory.yml and
re-renders the README's connector table. docs/PORTING.md says what to fill in next.

Usage:
    python scripts/new_connector.py confluence --display-name "Confluence"
    python scripts/new_connector.py apollo --display-name "Apollo" --auth api_key

The slug is the provider id: snake_case (``google_drive``), used for the directory,
the entry-point names and the ``<slug>_source`` factory. The module is
``cognee_connector_<slug>`` and the distribution ``cognee-connector-<slug-with-dashes>``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from packages import REPO_ROOT
from render_inventory import main as render_readme
from validate_inventory import DEFAULT_INVENTORY

TEMPLATE_DIR = REPO_ROOT / "templates" / "connector"
AUTH_CHOICES = ("oauth2", "api_key", "local")
# Marker comment lines in the templates that the generator replaces with auth-specific
# lines (or removes), keeping the templates valid Python that ruff can check.
MARKERS: dict[str, dict[str, list[str]]] = {
    "# __AUTH_FIELDS__": {
        "oauth2": [],
        "api_key": ['api_key: str = ""'],
        "local": ['root_path: str = ""'],
    },
    "# __OAUTH_FIELDS__": {
        "oauth2": [
            "# OAuth2 endpoints, without query strings; fixed query parameters go in",
            "# ``authorize_params``.",
            'authorize_url="https://example.com/oauth/authorize",',
            'token_url="https://example.com/oauth/token",',
            'scopes="",',
            "# Dotted paths into the token response: the external account id (what",
            "# webhooks route by) and a human-readable label for the UI.",
            'account_id_path="workspace.id",',
            'account_label_path="workspace.name",',
        ],
        "api_key": [],
        "local": [],
    },
}
AUTH_ENV_VARS = {
    "oauth2": ["CLIENT_ID", "CLIENT_SECRET", "REDIRECT_URI", "FRONTEND_BASE_URL"],
    "api_key": ["API_KEY"],
    "local": ["ROOT_PATH"],
}


def substitutions(slug: str, display_name: str, auth: str) -> dict[str, str]:
    # The slug names the directory and may use dashes (google-drive); the provider id
    # is its snake_case form, usable as a Python identifier and an env-var prefix.
    provider = slug.replace("-", "_")
    class_name = "".join(part.capitalize() for part in provider.split("_"))
    return {
        "__SLUG__": slug,
        "__PROVIDER__": provider,
        "__MODULE__": f"cognee_connector_{provider}",
        "__DIST__": "cognee-connector-" + provider.replace("_", "-"),
        "__DISPLAY_NAME__": display_name,
        "__AUTH__": auth,
        "__ENV_PREFIX__": provider.upper() + "_",
        "__CLASS__": class_name,
    }


def render(text: str, values: dict[str, str], auth: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        marker = line.strip()
        if marker in MARKERS:
            fields = MARKERS[marker][auth]
            if not fields and lines and not lines[-1].strip():
                # A marker that expands to nothing takes the blank line before it
                # along, so the generated block keeps ruff's blank-line count.
                lines.pop()
            indent = line[: len(line) - len(line.lstrip())]
            lines.extend(f"{indent}{field}\n" for field in fields)
            continue
        lines.append(line)
    rendered = "".join(lines)
    for token, value in values.items():
        rendered = rendered.replace(token, value)
    return rendered


def inventory_entry(values: dict[str, str], auth: str) -> str:
    display = values["__DISPLAY_NAME__"].replace("\\", "\\\\").replace('"', '\\"')
    env_vars = ", ".join(values["__ENV_PREFIX__"] + name for name in AUTH_ENV_VARS[auth])
    tier = "1" if auth == "oauth2" else "null"
    return (
        f"  - slug: {values['__SLUG__']}\n"
        f"    package: {values['__DIST__']}\n"
        f"    module: {values['__MODULE__']}\n"
        f"    directory: connectors/{values['__SLUG__']}\n"
        f'    display_name: "{display}"\n'
        f"    auth: {auth}\n"
        f"    tier: {tier}\n"
        f"    sources: [{values['__PROVIDER__']}]\n"
        "    hosts:\n"
        "      cognee: true\n"
        "      cloud: false\n"
        "    status: planned\n"
        "    origin:\n"
        "      cognee: null\n"
        "      cloud: null\n"
        f"    env_vars: [{env_vars}]\n"
    )


def append_inventory_entry(path: Path, entry: str) -> None:
    text = path.read_text(encoding="utf-8")
    empty = re.search(r"^connectors:\s*\[\]\s*$", text, flags=re.MULTILINE)
    if empty:
        text = text[: empty.start()] + "connectors:" + text[empty.end() :]
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + entry, encoding="utf-8")


def scaffold(slug: str, display_name: str, auth: str, root: Path = REPO_ROOT) -> Path:
    values = substitutions(slug, display_name, auth)
    destination = root / "connectors" / slug
    if destination.exists():
        raise SystemExit(f"error: {destination} already exists")
    if not TEMPLATE_DIR.is_dir():
        raise SystemExit(f"error: template directory {TEMPLATE_DIR} is missing")

    for source in sorted(TEMPLATE_DIR.rglob("*")):
        if source.is_dir():
            continue
        relative = source.relative_to(TEMPLATE_DIR)
        target = destination.joinpath(*(render(part, values, auth) for part in relative.parts))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render(source.read_text(encoding="utf-8"), values, auth), encoding="utf-8"
        )
        print(f"created {target.relative_to(root)}")

    shutil.copyfile(root / "LICENSE", destination / "LICENSE")
    print(f"created {(destination / 'LICENSE').relative_to(root)}")

    append_inventory_entry(DEFAULT_INVENTORY, inventory_entry(values, auth))
    print(f"appended {slug} to {DEFAULT_INVENTORY.relative_to(root)} (status: planned)")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scaffold a new connector package.")
    parser.add_argument(
        "slug", help="the provider id in snake_case, e.g. confluence or google_drive"
    )
    parser.add_argument(
        "--display-name", required=True, help='the name shown to users, e.g. "Confluence"'
    )
    parser.add_argument(
        "--auth", choices=AUTH_CHOICES, default="oauth2", help="how the provider authenticates"
    )
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[a-z][a-z0-9_-]*", args.slug):
        parser.error(
            "slug must be lowercase letters, digits, underscores or dashes, e.g. google-drive"
        )
    if not args.display_name.strip():
        parser.error("--display-name must not be empty")

    destination = scaffold(args.slug, args.display_name.strip(), args.auth)
    render_readme([])
    print(
        f"\nNext: fill in {destination.relative_to(REPO_ROOT)}/{substitutions(args.slug, '', args.auth)['__MODULE__']}/"
        "connector.py, settings.py and source.py (docs/PORTING.md), then run the tests with\n"
        f"  PYTHONPATH=core:{destination.relative_to(REPO_ROOT)} python -m pytest "
        f"{destination.relative_to(REPO_ROOT)}/tests -q"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

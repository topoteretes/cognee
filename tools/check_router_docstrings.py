#!/usr/bin/env python3
"""
Check that route-handler docstrings match the handler signatures.

For every route reachable from the FastAPI app in ``cognee.api.client`` (the
same set of routes the published OpenAPI spec is generated from), compare the
parameters documented in the handler docstring against the parameters the
handler actually accepts.

First version: parameters only.

Documented parameters live under the ``## Request Parameters`` /
``## Path Parameters`` / ``## Query Parameters`` docstring sections.
Two matching rules keep noise down:

- "undocumented" (real parameter never mentioned): a parameter counts as
  documented if its name appears anywhere inside a parameters section —
  as a ``- **name**`` bullet or in prose.
- "stale" (documented parameter that does not exist): only explicit
  ``- **name**`` bullets are held to this, since prose can mention anything.

Actual parameters are the handler signature parameters (minus FastAPI
dependencies and framework objects), with Pydantic body models expanded to
their field names, since docstrings document DTO fields rather than the
wrapper argument. Names are compared case-insensitively with underscores
stripped, so a ``datasetId`` bullet matches a ``dataset_id`` field (cognee
DTOs accept both spellings via the camelCase alias generator).

Exit codes: 0 = no issues, 1 = issues found, 2 = app import failed.
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import typing
from dataclasses import dataclass, field

# Markdown-style sections ("## Request Parameters") and Google-style ones ("Args:")
PARAM_SECTION_RE = re.compile(
    r"^\s{0,8}(#{2,4}\s+.*(parameters|request body).*|((query|path|request)\s+)?(args|arguments|parameters)\s*:)\s*$",
    re.IGNORECASE,
)
ANY_SECTION_RE = re.compile(
    r"^\s{0,8}(#{2,4}\s+\S|(returns|raises|yields|notes|examples|response|attributes)\s*:\s*$)",
    re.IGNORECASE,
)
BULLET_RE = re.compile(
    r"^\s*(?:-\s+\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*|([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s+\S)"
)
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize(name: str) -> str:
    """Case/underscore-insensitive form so snake_case matches camelCase aliases."""
    return name.replace("_", "").lower()


def documented_params(docstring: str | None) -> tuple[set[str], set[str]]:
    """(bullet names, all identifier tokens) found inside parameter sections."""
    if not docstring:
        return set(), set()

    bullets: set[str] = set()
    tokens: set[str] = set()
    in_param_section = False
    for line in docstring.splitlines():
        if PARAM_SECTION_RE.match(line):
            in_param_section = True
            continue
        if ANY_SECTION_RE.match(line):
            in_param_section = False
            continue
        if in_param_section:
            match = BULLET_RE.match(line)
            if match:
                bullets.add(match.group(1) or match.group(2))
            tokens.update(IDENTIFIER_RE.findall(line))
    return bullets, tokens


def _pydantic_models(annotation) -> list[type]:
    """Pydantic models mentioned in an annotation (unwraps Annotated/Optional/Union)."""
    from pydantic import BaseModel

    if annotation is inspect.Parameter.empty:
        return []
    origin = typing.get_origin(annotation)
    if origin is not None:
        models = []
        for arg in typing.get_args(annotation):
            models.extend(_pydantic_models(arg))
        return models
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return [annotation]
    return []


def _is_dependency(parameter: inspect.Parameter) -> bool:
    from fastapi import params as fastapi_params

    if isinstance(parameter.default, fastapi_params.Depends):
        return True
    if typing.get_origin(parameter.annotation) is typing.Annotated:
        return any(
            isinstance(meta, fastapi_params.Depends)
            for meta in typing.get_args(parameter.annotation)[1:]
        )
    return False


def _wire_alias(parameter: inspect.Parameter) -> str | None:
    """Alias set via Query/Form/Path/... — the name the client actually sends."""
    from fastapi import params as fastapi_params

    candidates = [parameter.default]
    if typing.get_origin(parameter.annotation) is typing.Annotated:
        candidates.extend(typing.get_args(parameter.annotation)[1:])
    for candidate in candidates:
        if isinstance(candidate, fastapi_params.Param) and candidate.alias:
            return candidate.alias
    return None


def actual_params(endpoint) -> tuple[set[str], set[str]]:
    """(names to document, normalized names acceptable in docs) for a handler.

    The first set holds one canonical wire name per client-facing parameter
    (Query/Form alias when set, else the signature name; DTO body models are
    expanded to their fields). The second, wider set also accepts unaliased
    signature names, both DTO field spellings, and DTO wrapper argument names,
    so documenting any legitimate spelling is not reported as stale.
    """
    import fastapi

    framework_types = (
        fastapi.Request,
        fastapi.Response,
        fastapi.BackgroundTasks,
        fastapi.WebSocket,
    )

    try:
        signature = inspect.signature(endpoint)
    except (TypeError, ValueError):
        return set(), set()

    canonical: set[str] = set()
    accepted_norms: set[str] = set()
    for name, parameter in signature.parameters.items():
        # Dependencies (auth user etc.) are not client-facing parameters,
        # but docstrings may still legitimately mention them — and for form
        # dependencies like OAuth2PasswordRequestForm, their own fields
        # (username, password, ...) are real wire fields worth documenting.
        if _is_dependency(parameter):
            accepted_norms.add(normalize(name))
            base = parameter.annotation
            if typing.get_origin(base) is typing.Annotated:
                base = typing.get_args(base)[0]
            if inspect.isclass(base):
                try:
                    dep_signature = inspect.signature(base.__init__)
                except (TypeError, ValueError):
                    dep_signature = None
                if dep_signature is not None:
                    for dep_name in dep_signature.parameters:
                        if dep_name != "self":
                            accepted_norms.add(normalize(dep_name))
            continue

        annotation = parameter.annotation
        base_annotation = annotation
        if typing.get_origin(annotation) is typing.Annotated:
            base_annotation = typing.get_args(annotation)[0]
        if inspect.isclass(base_annotation) and issubclass(base_annotation, framework_types):
            continue

        # Body DTOs: docstrings document the model fields, not the argument —
        # but referring to the wrapper argument is accepted too.
        models = _pydantic_models(annotation)
        if models:
            accepted_norms.add(normalize(name))
            for model in models:
                for field_name, model_field in model.model_fields.items():
                    canonical.add(model_field.alias or field_name)
                    accepted_norms.add(normalize(field_name))
                    if model_field.alias:
                        accepted_norms.add(normalize(model_field.alias))
            continue

        alias = _wire_alias(parameter)
        canonical.add(alias or name)
        accepted_norms.add(normalize(name))
        if alias:
            accepted_norms.add(normalize(alias))
    return canonical, accepted_norms


@dataclass
class RouteReport:
    method: str
    path: str
    endpoint_ref: str
    missing_docstring: bool = False
    undocumented: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)

    def issue_keys(self) -> list[str]:
        keys = []
        if self.missing_docstring:
            keys.append(f"{self.method} {self.path} :: missing-docstring :: -")
        keys.extend(
            f"{self.method} {self.path} :: undocumented :: {name}" for name in self.undocumented
        )
        keys.extend(f"{self.method} {self.path} :: stale :: {name}" for name in self.stale)
        return keys


def iter_api_routes(app):
    """Yield (path, APIRoute) for every route on the app, resolving deferred includes.

    FastAPI < 0.137 stores flat ``APIRoute`` objects on ``app.routes``. Newer
    versions defer ``include_router`` into ``_IncludedRouter`` placeholders whose
    ``effective_route_contexts()`` yield the materialized routes (full prefixed
    path + the original ``APIRoute``). Handle both, with ``original_router`` as a
    last-resort fallback.
    """
    from fastapi.routing import APIRoute

    for route in app.routes:
        if isinstance(route, APIRoute):
            yield route.path, route
            continue

        contexts_method = getattr(route, "effective_route_contexts", None)
        if callable(contexts_method):
            for context in contexts_method():
                original = getattr(context, "original_route", None)
                if isinstance(original, APIRoute):
                    yield getattr(context, "path", original.path), original
            continue

        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            for sub_route in getattr(original_router, "routes", []):
                if isinstance(sub_route, APIRoute):
                    yield sub_route.path, sub_route


def check_routes() -> list[RouteReport]:
    from cognee.api.client import app

    reports: list[RouteReport] = []
    seen: set[tuple[str, str]] = set()

    for path, route in iter_api_routes(app):
        # Routes hidden from the schema never reach the published OpenAPI spec.
        if not route.include_in_schema:
            continue
        endpoint = inspect.unwrap(route.endpoint)
        module = getattr(endpoint, "__module__", "") or ""
        # Skip handlers not defined in cognee (e.g. fastapi-users auth routes).
        if not module.startswith("cognee."):
            continue

        method = ",".join(sorted(route.methods or []))
        key = (method, path)
        if key in seen:
            continue
        seen.add(key)

        endpoint_ref = f"{module}.{getattr(endpoint, '__qualname__', endpoint)}"
        report = RouteReport(method=method, path=path, endpoint_ref=endpoint_ref)
        reports.append(report)

        docstring = inspect.getdoc(endpoint)
        if not docstring:
            report.missing_docstring = True
            continue

        canonical, accepted_norms = actual_params(endpoint)
        bullets, tokens = documented_params(docstring)

        canonical_by_norm: dict[str, str] = {}
        for name in canonical:
            canonical_by_norm.setdefault(normalize(name), name)
        mentioned_norms = {normalize(name) for name in bullets | tokens}

        report.undocumented = sorted(
            name for norm, name in canonical_by_norm.items() if norm not in mentioned_norms
        )
        report.stale = sorted(name for name in bullets if normalize(name) not in accepted_norms)

    reports.sort(key=lambda item: (item.path, item.method))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print the report but always exit 0",
    )
    args = parser.parse_args()

    # Avoid prod-only initialization, same as OpenAPI spec generation in CI.
    os.environ.setdefault("ENV", "dev")

    try:
        reports = check_routes()
    except Exception as exc:
        print(f"Failed to import cognee API app: {exc}", file=sys.stderr)
        return 2

    flagged = [report for report in reports if report.issue_keys()]
    for report in flagged:
        print(f"{report.method} {report.path}  ({report.endpoint_ref})")
        if report.missing_docstring:
            print("    docstring: MISSING")
        for name in report.undocumented:
            print(f"    parameter not documented:            {name}")
        for name in report.stale:
            print(f"    documented parameter does not exist: {name}")
        print()

    issue_count = sum(len(report.issue_keys()) for report in reports)
    print(
        f"Checked {len(reports)} route handlers: "
        f"{len(reports) - len(flagged)} clean, {len(flagged)} flagged, {issue_count} issues."
    )

    if flagged and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

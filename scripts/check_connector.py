"""Check an SDK data-source connector against the connector contract.

    python scripts/check_connector.py asana
    python scripts/check_connector.py trello --ported
    python scripts/check_connector.py obsidian --no-cloud --root /path/to/checkout

It reads source with ``ast`` and runs ruff; it imports nothing from the connector. A FAIL is a
broken rule. A WARN is a heuristic that needs a human look. Behaviour that only a run shows
(tombstones after a complete listing, every stats key on every exit path, a cursor that must
advance) stays with the tests. The exit status is 1 when any check FAILs.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

CONNECTORS = Path("cognee/tasks/ingestion/connectors")
UNIT_TESTS = Path("cognee/tests/unit/tasks/connectors")
INTEGRATION_TESTS = Path("cognee/tests/integration/tasks/connectors")
INGESTION = Path("cognee/modules/integrations/ingestion.py")
CONTRACT_TEST = Path("cognee/tests/unit/modules/integrations/test_ingestion_contract.py")
EXAMPLES_README = Path("examples/integrations/README.md")

# Importing the connectors package must not load these, so a connector imports them lazily.
LAZY_ONLY = {"dlt", "requests", "httpx"}
SECRET_NAMES = {"token", "access_token", "refresh_token", "bearer", "authorization", "apikey"}
SECRET_SUFFIXES = ("api_key", "secret", "password")
RESPONSE_ATTRS = {"text", "url", "content", "body", "headers"}
CLOCK_CALLS = {
    "datetime.now",
    "datetime.datetime.now",
    "datetime.utcnow",
    "datetime.datetime.utcnow",
    "time.time",
    "date.today",
}
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical"}
DOC_SECTIONS = {
    "usage": r"usage",
    "what is read": r"what is (read|ingested)",
    "incremental model": r"incremental|how sync",
    "deletion": r"delet|forget",
    "limits": r"limit",
    "privacy": r"privacy",
    "testing": r"testing",
}
HEADER = (
    r"# Adapted from topoteretes/cognee-community \(Apache-2\.0\)",
    r"# Modified to ship in the Cognee SDK",
)


class Report:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.passed = 0
        self.fails = 0
        self.warns = 0

    def check(self, ok: bool, rule: str, where: str, message: str, level: str = "FAIL") -> bool:
        if ok:
            self.passed += 1
            if self.verbose:
                print(f"PASS  {rule}")
            return True
        if level == "FAIL":
            self.fails += 1
        else:
            self.warns += 1
        print(f"{level:<5} {rule:<28} {where}  {message}")
        return False


@dataclass
class Ctx:
    root: Path
    provider: str
    report: Report
    cloud: bool
    core: frozenset[str]

    def rel(self, path: Path, line: int | None = None) -> str:
        text = str(path.relative_to(self.root)) if path.is_absolute() else str(path)
        return f"{text}:{line}" if line else text

    def check(self, ok: bool, rule: str, path: Path, message: str, *, line=None, level="FAIL"):
        return self.report.check(ok, rule, self.rel(path, line), message, level)

    def each(self, rule: str, path: Path, problems, level: str = "FAIL") -> None:
        """One finding per ``(line, message)`` problem, or a single pass when there is none."""
        clean = True
        for line, message in problems:
            clean = False
            self.report.check(False, rule, self.rel(path, line), message, level)
        if clean:
            self.report.check(True, rule, "", "")

    def parse(self, path: Path) -> ast.Module | None:
        """The syntax tree, or ``None`` for a missing file (the layout check reports that)."""
        if not path.is_file():
            return None
        try:
            return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            self.report.check(False, "parse", self.rel(path), f"cannot parse: {error}")
            return None


def core_dependencies(root: Path) -> frozenset[str]:
    """Import-style names of the packages in pyproject's ``[project].dependencies``."""
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    names = re.findall(r'^\s*"([A-Za-z0-9_.\-]+)', block.group(1), re.MULTILINE) if block else []
    return frozenset(name.lower().replace("-", "_") for name in names)


def dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def const(node: ast.AST | None, value) -> bool:
    return (
        isinstance(node, ast.Constant) and node.value == value and type(node.value) is type(value)
    )


def dict_get(node: ast.AST | None, key: str) -> ast.AST | None:
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if const(k, key):
                return v
    return None


def imports(nodes: list[ast.stmt] | ast.AST, *, module_level: bool) -> list[tuple[str, int]]:
    """Top-level names of absolute imports. ``module_level`` skips function and class bodies."""
    found = []

    def visit(statements) -> None:
        for node in statements:
            if isinstance(node, ast.Import):
                found.extend((a.name.split(".")[0], node.lineno) for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.append((node.module.split(".")[0], node.lineno))
            elif isinstance(node, ast.If) and dotted(node.test).endswith("TYPE_CHECKING"):
                continue
            elif module_level and isinstance(node, (ast.If, ast.Try, ast.With)):
                visit(node.body)
                visit(getattr(node, "orelse", []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)

    if module_level:
        visit(nodes.body)
    else:
        visit(node for node in ast.walk(nodes) if isinstance(node, (ast.Import, ast.ImportFrom)))
    return found


def third_party(names: list[tuple[str, int]]) -> list[str]:
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    return sorted({n for n, _ in names if n not in stdlib and n != "cognee"})


# ---------------------------------------------------------------- layout


def check_layout(ctx: Ctx) -> None:
    p = ctx.provider
    required = [
        CONNECTORS / f"{p}.py",
        CONNECTORS / f"{p}.md",
        UNIT_TESTS / f"test_{p}_source.py",
        INTEGRATION_TESTS / f"test_{p}_ingestion.py",
        INTEGRATION_TESTS / f"test_{p}_forget.py",
    ]
    for path in required:
        ctx.check((ctx.root / path).is_file(), "layout.file", path, "missing")
    for path in (
        (UNIT_TESTS / f"fake_{p}.py", UNIT_TESTS / f"test_{p}_client.py") if ctx.cloud else ()
    ):
        ctx.check(
            (ctx.root / path).is_file(),
            "layout.file",
            path,
            "missing (expected for a connector with its own HTTP client)",
            level="WARN",
        )
    readme = ctx.root / EXAMPLES_README
    if readme.is_file():
        name = re.escape(p).replace("_", "[ _-]?")
        ctx.check(
            re.search(name, readme.read_text(encoding="utf-8"), re.IGNORECASE) is not None,
            "layout.examples-readme",
            EXAMPLES_README,
            f"does not mention {p}",
        )


def check_exports(ctx: Ctx) -> None:
    path = CONNECTORS / "__init__.py"
    tree = ctx.parse(ctx.root / path)
    if tree is None:
        return
    p = ctx.provider
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == p
        for alias in node.names
    }
    exported = {
        const_.value
        for node in tree.body
        if isinstance(node, ast.Assign) and dotted(node.targets[0]) == "__all__"
        for const_ in ast.walk(node.value)
        if isinstance(const_, ast.Constant)
    }
    factory = f"{p}_source"
    ctx.check(factory in imported, "exports.import", path, f"does not import {factory}")
    ctx.check(factory in exported, "exports.all", path, f"{factory} is not in __all__")
    builder = f"build_{p}_service"
    connector = ctx.parse(ctx.root / CONNECTORS / f"{p}.py")
    if connector and any(
        isinstance(n, ast.FunctionDef) and n.name == builder for n in connector.body
    ):
        ctx.check(
            builder in imported and builder in exported,
            "exports.builder",
            path,
            f"{builder} is defined but not exported",
            level="WARN",
        )
    docstring = ast.get_docstring(tree) or ""
    ctx.check(
        re.search(re.escape(p).replace("_", "[ _-]?"), docstring, re.IGNORECASE) is not None,
        "exports.docstring",
        path,
        f"package docstring does not mention {p}",
        level="WARN",
    )


def check_provider_entry(ctx: Ctx) -> None:
    p = ctx.provider
    path = INGESTION
    tree = ctx.parse(ctx.root / path)
    if tree is None:
        return
    table = next(
        (
            node.value
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and dotted(node.targets[0] if isinstance(node, ast.Assign) else node.target)
            == "_PROVIDERS"
            and isinstance(node.value, ast.Dict)
        ),
        None,
    )
    entry = dict_get(table, p)
    if not ctx.check(
        isinstance(entry, ast.Call), "provider.entry", path, f'no _PROVIDERS["{p}"] entry'
    ):
        return
    kwargs = {k.arg: k.value for k in entry.keywords}
    prefix = kwargs["prefix"].value if isinstance(kwargs.get("prefix"), ast.Constant) else ""
    ctx.check(
        re.fullmatch(r"[a-z0-9_]{1,63}", str(prefix)) is not None,
        "provider.prefix",
        path,
        f"prefix {prefix!r} must be [a-z0-9_], 1 to 63 chars",
        line=entry.lineno,
    )
    module = f"cognee.tasks.ingestion.connectors.{p}"
    ctx.check(
        const(kwargs.get("source_module"), module),
        "provider.source-module",
        path,
        f"source_module must be {module!r}",
        line=entry.lineno,
    )
    ctx.check(
        const(kwargs.get("source_name"), f"{p}_source"),
        "provider.source-name",
        path,
        f"source_name must be {p + '_source'!r}",
        line=entry.lineno,
    )
    contract = ctx.root / CONTRACT_TEST
    tree = ctx.parse(contract) if contract.is_file() else None
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)} if tree else set()
    ctx.check(
        p in strings,
        "provider.pinned-name",
        CONTRACT_TEST,
        f'no test names "{p}": the resource name must be pinned',
    )
    ctx.check(
        any(
            isinstance(v, str) and re.fullmatch(rf"{re.escape(str(prefix))}_[0-9a-f]{{8,}}", v)
            for v in strings
        ),
        "provider.pinned-literal",
        CONTRACT_TEST,
        f"no literal {prefix}_<hash> in the test: never compute the expected name",
        level="WARN",
    )


# ---------------------------------------------------------------- connector module


def check_factory_signature(ctx: Ctx, path: Path, factory: ast.FunctionDef) -> None:
    args = factory.args
    positional = [a.arg for a in args.posonlyargs + args.args]
    defaults = dict(zip(reversed(positional), reversed(args.defaults)))
    defaults.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d})
    names = set(positional) | {a.arg for a in args.kwonlyargs}
    line = factory.lineno
    ctx.check(
        "resource_name" in names, "factory.resource-name", path, "no resource_name", line=line
    )
    default = defaults.get("resource_name")
    ctx.check(
        not (isinstance(default, ast.Constant) and isinstance(default.value, str)),
        "factory.resource-name-default",
        path,
        "a constant default shares one table and cursor between two accounts; "
        "require it or derive it from the account",
        line=line,
    )
    ctx.check("check_active" in names, "factory.check-active", path, "no check_active", line=line)
    ctx.check(
        "service" in names or "http_client" in names,
        "factory.service",
        path,
        "no service= (or http_client=) injection point",
        line=line,
        level="FAIL" if ctx.cloud else "WARN",
    )
    ctx.check(
        not positional,
        "factory.keyword-only",
        path,
        f"positional parameters {positional}; new connectors take keywords only",
        line=line,
        level="WARN",
    )
    ctx.check(
        not ctx.cloud or any(re.match(r"max_|.*(budget|quota)", n) for n in names),
        "factory.budget",
        path,
        "no request budget kwarg (max_requests, a quota)",
        line=line,
        level="WARN",
    )
    ctx.check(
        any(
            isinstance(n, ast.Raise)
            and isinstance(n.exc, ast.Call)
            and dotted(n.exc.func) == "ValueError"
            for n in ast.walk(factory)
        ),
        "factory.fails-early",
        path,
        "no ValueError at construction (missing credentials or selection)",
        line=line,
        level="WARN",
    )
    ctx.check(
        any(name == "dlt" for name, _ in imports(factory, module_level=False)),
        "factory.lazy-dlt",
        path,
        "import dlt inside the factory",
        line=line,
    )


def sync_version_guard(factory: ast.FunctionDef) -> int | None:
    for node in ast.walk(factory):
        if not (isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.Lt)):
            continue
        mentions = any(
            const(n, "DOCUMENT_SYNC_VERSION")
            or (isinstance(n, ast.Name) and n.id == "DOCUMENT_SYNC_VERSION")
            or (isinstance(n, ast.Attribute) and n.attr == "DOCUMENT_SYNC_VERSION")
            for n in ast.walk(node.left)
        )
        right = node.comparators[0]
        if mentions and isinstance(right, ast.Constant) and isinstance(right.value, int):
            return right.value
    return None


def required_sync_version(tree: ast.Module) -> int:
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    if "STRUCTURE_COLUMN" in names or "cognee_structure" in strings:
        return 3
    if "NODE_SET_COLUMN" in names or "cognee_node_set" in strings:
        return 2
    return 1


def check_resource(ctx: Ctx, path: Path, tree: ast.Module, factory: ast.FunctionDef) -> None:
    p = ctx.provider
    line = factory.lineno
    guard = sync_version_guard(factory)
    needed = required_sync_version(tree)
    ctx.check(
        guard is not None and guard >= needed,
        "resource.sync-version",
        path,
        f"DOCUMENT_SYNC_VERSION guard must refuse < {needed} (found {guard})",
        line=line,
    )
    decorator = next(
        (
            dec
            for node in ast.walk(factory)
            if isinstance(node, ast.FunctionDef)
            for dec in node.decorator_list
            if isinstance(dec, ast.Call) and dotted(dec.func) == "dlt.resource"
        ),
        None,
    )
    if not ctx.check(decorator is not None, "resource.decorator", path, "no @dlt.resource(...)"):
        return
    kw = {k.arg: k.value for k in decorator.keywords}
    where = decorator.lineno
    ctx.check("name" in kw, "resource.name", path, "no name=", line=where)
    ctx.check(
        const(kw.get("primary_key"), "id"), "resource.primary-key", path, 'must be "id"', line=where
    )
    ctx.check(
        const(kw.get("write_disposition"), "merge"),
        "resource.write-disposition",
        path,
        'must be "merge"',
        line=where,
    )
    ctx.check(
        const(dict_get(dict_get(kw.get("columns"), "_deleted"), "hard_delete"), True),
        "resource.hard-delete",
        path,
        'columns must hold {"_deleted": {..., "hard_delete": True}}',
        line=where,
    )
    sets = [
        (dotted(n.args[1]), n.args[2], n.lineno)
        for n in ast.walk(factory)
        if isinstance(n, ast.Call) and dotted(n.func) == "setattr" and len(n.args) == 3
    ]
    source = next((v for k, v, _ in sets if k.endswith("DOCUMENT_SOURCE_ATTR")), None)
    ctx.check(
        const(source, p),
        "resource.document-source",
        path,
        f"DOCUMENT_SOURCE_ATTR must be set to {p!r}, the _PROVIDERS key",
        line=line,
    )
    scope = next((v for k, v, _ in sets if k.endswith("PIPELINE_SCOPE_ATTR")), None)
    name_kw = kw.get("name")
    ctx.check(
        isinstance(scope, ast.Name) and isinstance(name_kw, ast.Name) and scope.id == name_kw.id,
        "resource.pipeline-scope",
        path,
        "PIPELINE_SCOPE_ATTR must be set to the same name the resource is declared with",
        line=line,
    )
    ctx.check(
        any(
            isinstance(n, ast.Assign)
            and any(
                dotted(t) == "resource.cognee_sync_stats"
                or (isinstance(t, ast.Attribute) and t.attr == "cognee_sync_stats")
                for t in n.targets
            )
            for n in ast.walk(factory)
        ),
        "resource.sync-stats",
        path,
        "cognee_sync_stats is never set on the resource",
        line=line,
    )
    ctx.check(
        any(const(n, "failed") for n in ast.walk(tree)),
        "resource.failed-key",
        path,
        'no "failed" stats key',
    )
    calls = {dotted(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    ctx.check(
        "dlt.current.resource_state" in calls,
        "resource.state",
        path,
        "incremental state must live in dlt.current.resource_state()",
    )
    ctx.check(
        any(c.endswith("guarded_rows") for c in calls),
        "resource.guarded-rows",
        path,
        "rows are not wrapped in dlt_utils.guarded_rows(...)",
        level="WARN",
    )
    ctx.check(
        any(const(n, "v") for n in ast.walk(tree)),
        "resource.state-version",
        path,
        'no state["v"] version key',
        level="WARN",
    )


def message_exprs(call: ast.Call) -> Iterator[ast.AST]:
    """Expressions that end up in a message: f-string values and non-literal arguments."""
    for arg in [*call.args, *(k.value for k in call.keywords)]:
        for node in ast.walk(arg):
            if isinstance(node, ast.FormattedValue):
                yield node.value
        if not isinstance(arg, (ast.Constant, ast.JoinedStr)):
            yield arg


def message_problems(tree: ast.Module) -> Iterator[tuple[str, int, str]]:
    """``(rule, line, text)`` for what a raised error or a log line would leak.

    A caught exception formatted whole is flagged in a raise only: a library error carries a URL
    into the new error, while the log lines of a typed error of the connector are safe by contract.
    """
    caught = {n.name for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.name}
    messages = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            messages.append((node.exc, True))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in LOG_METHODS
            and dotted(node.func.value).split(".")[0] in {"logger", "log", "_logger", "LOGGER"}
        ):
            messages.append((node, False))
    for call, raising in messages:
        for expr in message_exprs(call):
            safe = {id(n.value) for n in ast.walk(expr) if isinstance(n, ast.Attribute)}
            safe |= {
                id(n.args[0])
                for n in ast.walk(expr)
                if isinstance(n, ast.Call) and dotted(n.func) == "type" and n.args
            }
            for node in ast.walk(expr):
                if isinstance(node, ast.Name):
                    ident = node.id
                elif isinstance(node, ast.Attribute):
                    ident = node.attr
                else:
                    continue
                if ident.lower() in SECRET_NAMES or ident.lower().endswith(SECRET_SUFFIXES):
                    yield "messages.secret", node.lineno, f"{ident!r} is formatted into a message"
                elif isinstance(node, ast.Attribute) and ident in RESPONSE_ATTRS:
                    yield (
                        "messages.response",
                        node.lineno,
                        f".{ident} (response text, url or headers) is formatted into a message",
                    )
                elif (
                    raising
                    and isinstance(node, ast.Name)
                    and ident in caught
                    and id(node) not in safe
                ):
                    yield (
                        "messages.caught-exception",
                        node.lineno,
                        (
                            f"formats the caught exception {ident!r}; a library error can "
                            "carry a URL. Map it to a typed error with `from None`"
                        ),
                    )


def check_messages(ctx: Ctx, path: Path, tree: ast.Module) -> None:
    problems = list(message_problems(tree))
    for rule, level in (
        ("messages.secret", "FAIL"),
        ("messages.response", "WARN"),
        ("messages.caught-exception", "WARN"),
    ):
        ctx.each(rule, path, [(ln, text) for r, ln, text in problems if r == rule], level)


def check_module(ctx: Ctx, path: Path, tree: ast.Module) -> None:
    p = ctx.provider
    allowed = set(sys.stdlib_module_names) | {"__future__", "cognee"} | (ctx.core - LAZY_ONLY)
    ctx.each(
        "module.lazy-imports",
        path,
        [
            (line, f"imports {name} at module level; import it inside the function that uses it")
            for name, line in imports(tree, module_level=True)
            if name not in allowed
        ],
    )
    extras = [
        name
        for name in third_party(imports(tree, module_level=False))
        if name not in ctx.core | LAZY_ONLY
    ]
    ctx.check(
        not extras,
        "module.dependencies",
        path,
        f"third-party imports {extras}: confirm the pyproject extra and uv.lock",
        level="WARN",
    )
    nodes = list(ast.walk(tree))
    ctx.each(
        "module.env",
        path,
        [
            (n.lineno, "reads the environment (no env fallbacks)")
            for n in nodes
            if (isinstance(n, ast.Call) and dotted(n.func) == "os.getenv")
            or (isinstance(n, ast.Attribute) and dotted(n) == "os.environ")
        ],
    )
    ctx.each(
        "module.global",
        path,
        [
            (n.lineno, "global statement (no module state)")
            for n in nodes
            if isinstance(n, ast.Global)
        ],
    )
    ctx.each(
        "module.local-clock",
        path,
        [
            (
                n.lineno,
                "local clock: a cursor or rendered text taken from it skips changes or re-cognifies",
            )
            for n in nodes
            if isinstance(n, ast.Call) and dotted(n.func) in CLOCK_CALLS
        ],
        "WARN",
    )
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            mutable = isinstance(node.value, (ast.List, ast.Dict, ast.Set)) or (
                isinstance(node.value, ast.Call)
                and dotted(node.value.func) in {"dict", "list", "set", "defaultdict"}
            )
            ctx.check(
                not (mutable and target != target.upper()),
                "module.state",
                path,
                f"module-level mutable {target!r}",
                line=node.lineno,
                level="WARN",
            )
    keywords = [k for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords]
    ctx.check(
        not any(k.arg == "verify" and const(k.value, False) for k in keywords),
        "http.verify",
        path,
        "TLS verification is switched off",
    )
    ctx.check(
        not any(
            k.arg in {"allow_redirects", "follow_redirects"} and const(k.value, True)
            for k in keywords
        ),
        "http.redirects",
        path,
        "follows redirects: a 3xx must be an error",
    )
    modules = {name for name, _ in imports(tree, module_level=False)}
    ctx.check(
        "requests" not in modules
        or any(k.arg == "allow_redirects" and const(k.value, False) for k in keywords),
        "http.redirect-policy",
        path,
        "uses requests without allow_redirects=False (it follows redirects by default)",
        level="WARN",
    )
    holders = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(
            a.arg in {"token", "access_token", "api_key"}
            for method in node.body
            if isinstance(method, ast.FunctionDef)
            for a in [*method.args.args, *method.args.kwonlyargs]
        )
        and not any(isinstance(m, ast.FunctionDef) and m.name == "__repr__" for m in node.body)
    ]
    ctx.check(
        not holders,
        "errors.repr",
        path,
        f"{holders} take a token or key but define no redacting __repr__",
        level="WARN",
    )
    ctx.check(
        any(
            isinstance(n, ast.ClassDef)
            and any(dotted(b).endswith(("Error", "Exception")) for b in n.bases)
            for n in ast.walk(tree)
        ),
        "errors.typed",
        path,
        "no typed error class",
        level="WARN",
    )
    ctx.check(
        any(isinstance(n, ast.Raise) and const(n.cause, None) for n in ast.walk(tree)),
        "errors.from-none",
        path,
        "no `raise ... from None` to drop a URL-bearing library message",
        level="WARN",
    )
    loggers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and dotted(n.func).endswith("getLogger")
    ]
    named = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and dotted(n.func).endswith("get_logger")
    ]
    ctx.check(
        not loggers and any(n.args and const(n.args[0], f"{p}_connector") for n in named),
        "module.logger",
        path,
        f'use cognee.shared.logging_utils.get_logger("{p}_connector")',
        level="WARN",
    )
    check_messages(ctx, path, tree)


def check_connector(ctx: Ctx) -> None:
    path = CONNECTORS / f"{ctx.provider}.py"
    if not (ctx.root / path).is_file():
        return
    tree = ctx.parse(ctx.root / path)
    if tree is None:
        return
    check_module(ctx, path, tree)
    factory = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == f"{ctx.provider}_source"
        ),
        None,
    )
    if ctx.check(factory is not None, "factory.exists", path, f"no {ctx.provider}_source()"):
        check_factory_signature(ctx, path, factory)
        check_resource(ctx, path, tree, factory)


# ---------------------------------------------------------------- docs, tests, ruff


def check_doc(ctx: Ctx) -> None:
    path = CONNECTORS / f"{ctx.provider}.md"
    if not (ctx.root / path).is_file():
        return
    headings = re.findall(
        r"^#+\s+(.*)$", (ctx.root / path).read_text(encoding="utf-8"), re.MULTILINE
    )
    for section, pattern in DOC_SECTIONS.items():
        ctx.check(
            any(re.search(pattern, h, re.IGNORECASE) for h in headings),
            "doc.section",
            path,
            f"no section for: {section}",
            level="WARN",
        )


def related_files(ctx: Ctx) -> list[Path]:
    """The connector and every test file named after it, relative to the root."""
    p = ctx.provider
    found = [CONNECTORS / f"{p}.py"]
    for directory in (UNIT_TESTS, INTEGRATION_TESTS):
        for pattern in (f"test_{p}_*.py", f"fake_{p}.py"):
            found += [f.relative_to(ctx.root) for f in sorted((ctx.root / directory).glob(pattern))]
    return found


def check_ported_header(ctx: Ctx, files: list[Path]) -> None:
    for path in files:
        if not (ctx.root / path).is_file():
            continue
        lines = (ctx.root / path).read_text(encoding="utf-8").splitlines()[:2]
        ok = len(lines) == 2 and all(re.match(h, line) for h, line in zip(HEADER, lines))
        ctx.check(ok, "ported.header", path, "missing the two-line cognee-community attribution")


def check_tests(ctx: Ctx, files: list[Path]) -> None:
    client = UNIT_TESTS / f"test_{ctx.provider}_client.py"
    if (ctx.root / client).is_file():
        tree = ctx.parse(ctx.root / client)
        if tree is not None:
            ints = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
            for status in (401, 403, 404, 429):
                ctx.check(
                    status in ints,
                    "tests.statuses",
                    client,
                    f"no case for HTTP {status}",
                    level="WARN",
                )
            ctx.check(
                any(isinstance(i, int) and 500 <= i <= 599 for i in ints),
                "tests.statuses",
                client,
                "no case for a 5xx",
                level="WARN",
            )
            ctx.check(
                any(
                    isinstance(n, ast.Compare) and any(isinstance(o, ast.NotIn) for o in n.ops)
                    for n in ast.walk(tree)
                ),
                "tests.token-not-in-message",
                client,
                "no `not in` assertion: the key, id and URL must be absent from errors and logs",
                level="WARN",
            )
    texts = [
        (ctx.root / f).read_text(encoding="utf-8")
        for f in files
        if f.name.startswith("test_") and (ctx.root / f).is_file()
    ]
    ctx.check(
        any(re.search(r"shuffle|reversed\(|random\.", t) for t in texts),
        "tests.determinism",
        UNIT_TESTS / f"test_{ctx.provider}_source.py",
        "no test shuffles collections to prove the rendering is deterministic",
        level="WARN",
    )


def find_ruff(root: Path, explicit: str | None) -> list[str] | None:
    if explicit:
        return [explicit]
    for base in (root, Path(__file__).resolve().parents[1]):
        local = base / ".venv" / "bin" / "ruff"
        if local.is_file():
            return [str(local)]
    found = shutil.which("ruff")
    return [found] if found else None


def check_ruff(ctx: Ctx, files: list[Path], explicit: str | None) -> None:
    ruff = find_ruff(ctx.root, explicit)
    existing = [str(f) for f in files if (ctx.root / f).is_file()]
    if ruff is None:
        print("SKIP  ruff                         not found; pass --ruff or --skip-ruff")
        return
    runs = {
        "ruff.check": ["check"],
        "ruff.strict-gate": [
            "check",
            "--select",
            "BLE001,S110,S112,G201,TRY201,TRY002",
            "--ignore-noqa",
        ],
        "ruff.format": ["format", "--check"],
    }
    for rule, args in runs.items():
        proc = subprocess.run(
            [*ruff, *args, "--no-cache", *existing],
            cwd=ctx.root,
            capture_output=True,
            text=True,
            check=False,
        )
        output = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[:15])
        ctx.check(proc.returncode == 0, rule, Path("."), f"\n{output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("provider", help="provider key, e.g. asana or google_drive")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ported", action="store_true", help="require the community header")
    parser.add_argument(
        "--no-cloud", action="store_true", help="no _PROVIDERS entry (local source)"
    )
    parser.add_argument("--skip-ruff", action="store_true")
    parser.add_argument("--ruff", help="path to the ruff binary")
    parser.add_argument("-v", "--verbose", action="store_true", help="also print passing checks")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.provider):
        parser.error("provider must be lowercase letters, digits and underscores")

    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file():
        parser.error(f"{root} has no pyproject.toml; pass --root a cognee checkout")
    ctx = Ctx(root, args.provider, Report(args.verbose), not args.no_cloud, core_dependencies(root))
    files = related_files(ctx)
    check_layout(ctx)
    check_exports(ctx)
    if not args.no_cloud:
        check_provider_entry(ctx)
    check_connector(ctx)
    check_doc(ctx)
    check_tests(ctx, files)
    if args.ported:
        check_ported_header(ctx, files)
    if not args.skip_ruff:
        check_ruff(ctx, files, args.ruff)

    report = ctx.report
    print(
        f"\n{args.provider}: {report.passed} passed, {report.fails} failed, {report.warns} warnings"
    )
    return 1 if report.fails else 0


if __name__ == "__main__":
    sys.exit(main())

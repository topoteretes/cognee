"""Contract tests for the ``CogneeApiError`` exception hierarchy.

Regression guard for https://github.com/topoteretes/cognee/issues/3749, where
``EntityNotFoundError`` and ``NodesetFilterNotSupportedError`` (and, with the
same root cause, ``WrongTaskTypeError``) overrode ``__init__`` but never called
``super().__init__()``. That left ``Exception.args`` empty, breaking ``repr``,
exception chaining, and centralized logging.

Two complementary checks keep the whole bug class from returning:

* ``test_no_cognee_error_subclass_bypasses_base_init`` — a **static** AST sweep
  of every ``.py`` file in the package. It needs no imports, so it also covers
  subclasses that live outside ``*/exceptions/`` packages or inside modules that
  require optional extras (e.g. ``code_graph``, ``web_scraper``, ``neptune``).
* ``test_cognee_error_subclasses_preserve_args_and_chaining`` — a **runtime**
  check that instantiates every importable subclass and verifies the observable
  contract (``args`` populated, ``str`` non-empty, ``__cause__`` preserved).
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

import cognee
from cognee.exceptions import CogneeApiError


# Root classes of the Cognee error hierarchy. Any class that (transitively)
# subclasses one of these must run a base ``__init__``.
FAMILY_ROOTS = {
    "CogneeApiError",
    "CogneeSystemError",
    "CogneeValidationError",
    "CogneeConfigurationError",
    "CogneeTransientError",
}

PACKAGE_ROOT = Path(cognee.__file__).parent

# Representative values for constructor arguments that have no default, so we can
# instantiate every subclass regardless of its signature. A raw KeyError here is
# turned into an actionable failure (see ``_build_kwargs``).
SAMPLE_ARGUMENTS = {
    "attribute": "sample_attribute",
    "detail": "sample detail",
    "dimension": 1,
    "field": "sample_field",
    "got": "sample",
    "max_index": 2,
    "message": "contract message",
    "name": "ContractError",
    "observer": "sample-observer",
    "provider": "sample-provider",
    "search_type": "sample-search",
    "status_code": 400,
    "value": 1,
}


def _iter_source_files():
    """Yield every non-test, non-cache ``.py`` file in the ``cognee`` package."""
    for path in PACKAGE_ROOT.rglob("*.py"):
        parts = path.relative_to(PACKAGE_ROOT).parts
        if "__pycache__" in parts or "tests" in parts:
            continue
        yield path


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    return ".".join(relative.parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _base_names(class_node: ast.ClassDef) -> set:
    """Names of a class's bases, covering both ``Base`` and ``module.Base`` forms."""
    names = set()
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _family_classes_in_repo():
    """Statically discover the family and where each class is defined.

    Resolution is by base-class *name* and iterated to a fixpoint, so multi-level
    subclasses (a class extending an intermediate subclass defined elsewhere) are
    picked up too — no imports required. Both ``Base`` and dotted ``module.Base``
    references are recognised.

    Returns ``(classes, family_names)`` where ``classes`` is a list of
    ``(path, ClassDef)`` and ``family_names`` is the resolved set of every class
    name in the hierarchy (used to validate base ``__init__`` calls precisely).
    """
    definitions = []  # (module_path, ClassDef, {base names})
    for path in _iter_source_files():
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ClassDef):
                base_names = _base_names(node)
                if base_names:
                    definitions.append((path, node, base_names))

    family_names = set(FAMILY_ROOTS)
    changed = True
    while changed:
        changed = False
        for _path, node, base_names in definitions:
            if node.name not in family_names and (base_names & family_names):
                family_names.add(node.name)
                changed = True

    classes = [
        (path, node) for path, node, base_names in definitions if (base_names & family_names)
    ]
    return classes, family_names


def _defines_init_without_base_call(class_node: ast.ClassDef, family_names: set) -> bool:
    """True if the class overrides ``__init__`` but never calls a *family* base ``__init__``.

    A call only counts when its target is ``super()`` or a class in the Cognee error
    hierarchy (``FamilyBase.__init__`` / ``module.FamilyBase.__init__``). Calling an
    unrelated ``__init__`` — e.g. the grandparent ``Exception.__init__`` or a mixin's —
    does NOT satisfy the contract, because it skips ``CogneeApiError.__init__`` (centralized
    logging + status defaults). A class that does not override ``__init__`` inherits the
    base and is fine.
    """
    init = next(
        (n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    if init is None:
        return False
    for node in ast.walk(init):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "__init__"
        ):
            continue
        target = node.func.value
        # super().__init__(...)
        if (
            isinstance(target, ast.Call)
            and isinstance(target.func, ast.Name)
            and target.func.id == "super"
        ):
            return False
        # FamilyBase.__init__(self, ...) or module.FamilyBase.__init__(self, ...)
        if isinstance(target, ast.Name) and target.id in family_names:
            return False
        if isinstance(target, ast.Attribute) and target.attr in family_names:
            return False
    return True


def test_no_cognee_error_subclass_bypasses_base_init():
    """Static guard: no error subclass may skip its ``CogneeApiError`` base ``__init__``."""
    family, family_names = _family_classes_in_repo()
    # Sanity: the sweep actually found the hierarchy (guards against a silently
    # empty match, e.g. if discovery ever breaks).
    assert len(family) > 20, f"Only found {len(family)} family classes; discovery likely broke."

    offenders = [
        f"{path.relative_to(PACKAGE_ROOT.parent)}:{node.lineno} {node.name}"
        for path, node in family
        if _defines_init_without_base_call(node, family_names)
    ]
    assert not offenders, "CogneeError subclasses missing super().__init__():\n" + "\n".join(
        offenders
    )


def _import_family_modules():
    """Import every module that defines a family class, tolerating optional extras."""
    classes, _family_names = _family_classes_in_repo()
    module_paths = {path for path, _node in classes}
    for path in module_paths:
        try:
            importlib.import_module(_module_name(path))
        except Exception:
            # Modules behind optional extras (codegraph, scraping, neptune, ...)
            # may not import in a minimal environment. The static test above
            # already covers them; here we simply skip what we cannot load.
            continue


def _all_subclasses(root):
    seen = set()
    stack = list(root.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return seen


def _build_kwargs(cls):
    kwargs = {}
    for name, parameter in inspect.signature(cls).parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.default is inspect.Parameter.empty:
            if name not in SAMPLE_ARGUMENTS:
                pytest.fail(
                    f"{cls.__module__}.{cls.__name__} has required argument '{name}' with no "
                    f"sample value; add one to SAMPLE_ARGUMENTS in this test."
                )
            kwargs[name] = SAMPLE_ARGUMENTS[name]
    return kwargs


def test_cognee_error_subclasses_preserve_args_and_chaining():
    """Runtime contract for every importable subclass in the hierarchy."""
    _import_family_modules()

    classes = _all_subclasses(CogneeApiError)
    assert classes, "No CogneeApiError subclasses discovered at runtime."

    for cls in classes:
        exc = cls(**_build_kwargs(cls))

        # CogneeApiError.__init__ ran -> it forwards (message, name) to
        # Exception.__init__, so args must be exactly that pair. An empty args
        # (the original #3749 bug) or a differently-shaped args both fail here.
        assert exc.args == (exc.message, exc.name), (
            f"{cls.__name__}.args must be (message, name) from CogneeApiError.__init__; "
            f"got {exc.args!r}."
        )
        assert str(exc), f"{cls.__name__} has an empty str()."
        assert getattr(exc, "message", None), f"{cls.__name__} did not set .message."

        cause = RuntimeError("root cause")
        try:
            raise exc from cause
        except cls as raised:
            assert raised.__cause__ is cause, f"{cls.__name__} dropped exception chaining."

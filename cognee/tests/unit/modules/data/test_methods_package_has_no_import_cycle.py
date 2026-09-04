"""`cognee.modules.data.methods` must stay importable without a cycle.

Re-exporting a submodule from this package's ``__init__`` is only safe while
that submodule imports nothing that leads back here. When one did, the cycle
was invisible to every existing test and to a full-suite run, because they all
reach the package directly: ``__init__`` then runs to completion and every
name resolves to the function it should.

The failure needed the *other* order. ``cognee.add`` pulls in
``cognee.modules.pipelines.layers`` first, which does
``from cognee.modules.data.methods import load_or_create_datasets`` while this
package's ``__init__`` is still part-way through its own imports. At that
moment the function does not exist yet, so Python falls back to binding the
name to the same-named *submodule* -- and never rebinds it once ``__init__``
finishes. The call site then raised ``TypeError: 'module' object is not
callable`` on every add, while the package itself looked perfectly fine.

So these import in that order, in a subprocess with a clean module table:
importing them here would prove nothing, since pytest has already imported
half the tree.
"""

import subprocess
import sys
import textwrap

# Every name this package re-exports that a cycle could silently turn into a
# module. Kept explicit rather than derived from __all__: the point is to
# assert against a written-down expectation, not against whatever the package
# currently happens to expose.
REEXPORTED_CALLABLES = (
    "load_or_create_datasets",
    "get_authorized_existing_datasets",
    "check_dataset_name",
)


def _probe(source: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout.strip()


def test_pipelines_layers_first_still_binds_functions_not_modules():
    """The import order `cognee.add` actually uses must not degrade a name."""
    names = ", ".join(repr(n) for n in REEXPORTED_CALLABLES)
    out = _probe(
        f"""
        import inspect

        # The order that broke it: the pipelines layer resolves first, so it
        # reaches into data.methods while that package is mid-initialisation.
        import cognee.modules.pipelines.layers.resolve_authorized_user_datasets as layer
        import cognee.modules.data.methods as methods

        bad = []
        for name in ({names},):
            for holder, label in ((layer, "layer"), (methods, "package")):
                obj = getattr(holder, name, None)
                if obj is not None and inspect.ismodule(obj):
                    bad.append(f"{{label}}.{{name}}")
        print(",".join(bad) if bad else "clean")
        """
    )
    assert out == "clean", f"bound to a module instead of a function: {out}"


def test_package_import_alone_is_still_clean():
    """The direct order has always worked; keep it covered so a fix can't
    trade one for the other."""
    names = ", ".join(repr(n) for n in REEXPORTED_CALLABLES)
    out = _probe(
        f"""
        import inspect
        import cognee.modules.data.methods as methods

        missing = [n for n in ({names},) if not callable(getattr(methods, n, None))]
        print(",".join(missing) if missing else "clean")
        """
    )
    assert out == "clean", f"not callable when imported directly: {out}"

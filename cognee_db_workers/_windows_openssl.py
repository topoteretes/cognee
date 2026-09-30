"""Windows-only shim that makes ladybug's native extension importable.

Why it exists
-------------
ladybug's Windows wheels stopped vendoring OpenSSL in 0.19.0 — they are
repaired with ``delvewheel repair --exclude libssl-3-x64.dll --exclude
libcrypto-3-x64.dll`` — while ``ladybug/_lbug*.pyd`` still lists
``libssl-3-x64.dll`` and ``libcrypto-3-x64.dll`` in its import table. Since
Python 3.8 an extension module's dependencies are resolved through the secure
DLL search path (``PATH`` is not consulted), and nothing on that path carries
those two names, so ``import ladybug._lbug`` raises ImportError.
``ladybug._backend.get_pybind_module()`` swallows that error and returns None,
which makes ``ladybug.Database`` silently select its C-API backend instead —
and that backend's shared library is shipped in no wheel at all. The failure
therefore surfaces at the first database open as the unrelated::

    RuntimeError: Could not find lbug C API shared library.

What it does
------------
CPython on Windows ships the very same OpenSSL 3 libraries for its own ``_ssl``
module (3.11 onward), but under the unsuffixed names ``libssl-3.dll`` /
``libcrypto-3.dll``. This copies them into a cache directory under the names
ladybug's import table asks for, then registers that directory with
``os.add_dll_directory()`` — the only mechanism Windows offers for this.

``libssl-3-x64.dll`` is a byte copy of ``libssl-3.dll``, so its own import
table still names ``libcrypto-3.dll``; that name is placed in the cache
directory too, otherwise the renamed library cannot resolve its dependency.
ladybug consequently maps two instances of the same libcrypto build — the
``-x64`` one it calls directly, and the one behind libssl. They are used
independently (ladybug's TLS objects never leave libssl), so they do not
interact.

Lifetime
--------
Delete this module and its call sites once ladybug ships Windows wheels that
vendor OpenSSL again — the shim already turns itself off when the installed
wheel carries its own ``libssl-3-x64*.dll``. Tracked in COG-6185.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile

# What ladybug's extension imports -> the name CPython ships that library under.
# ``libcrypto-3.dll`` maps to itself: it is not imported by ladybug, it is the
# dependency of the renamed libssl copy.
_REQUIRED_DLLS = {
    "libssl-3-x64.dll": "libssl-3.dll",
    "libcrypto-3-x64.dll": "libcrypto-3.dll",
    "libcrypto-3.dll": "libcrypto-3.dll",
}


def ensure_ladybug_openssl() -> None:
    """Put the OpenSSL DLLs ladybug's extension needs on the DLL search path.

    A no-op off Windows, and after the first run on Windows it costs three
    ``os.path.exists`` calls. Must be called before ``ladybug`` is imported,
    and once per process — a DLL directory registered in the parent is not
    inherited by a spawned worker.
    """
    if sys.platform != "win32":
        return
    if _ladybug_vendors_openssl():
        return

    source_directory = os.path.join(sys.base_prefix, "DLLs")
    sources = {
        target: os.path.join(source_directory, source) for target, source in _REQUIRED_DLLS.items()
    }
    if not all(os.path.isfile(path) for path in sources.values()):
        # Python 3.10 links OpenSSL 1.1 (``libssl-1_1.dll``) and embedded
        # distributions ship no ``DLLs`` directory at all. There is nothing
        # ABI-compatible to hand ladybug, so leave the search path untouched
        # and let it report its own failure.
        return

    cache_directory = _cache_directory()
    os.makedirs(cache_directory, exist_ok=True)
    for target, source in sources.items():
        _copy_once(source, os.path.join(cache_directory, target))

    # The returned handle is deliberately dropped: CPython unregisters the
    # directory only on an explicit ``close()``, so the registration lives as
    # long as the process — which is what every later ``import ladybug`` needs.
    os.add_dll_directory(cache_directory)


def _cache_directory() -> str:
    """Where the renamed copies live, keyed by interpreter.

    Two environments on one machine can run different CPython builds; giving
    each its own directory keeps one from handing the other a mismatched
    OpenSSL.
    """
    tag = hashlib.sha256(sys.base_prefix.encode("utf-8")).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"cognee-ladybug-openssl-{tag}")


def _copy_once(source: str, target: str) -> None:
    """Copy ``source`` to ``target`` unless it is already there.

    Staged through a temporary file in the same directory and moved into place
    with ``os.replace`` so concurrently starting workers can never observe — or
    load — a half-written DLL.
    """
    if os.path.exists(target):
        return

    handle, staged = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".part")
    os.close(handle)
    try:
        shutil.copyfile(source, staged)
        os.replace(staged, target)
    except OSError:
        # Another process populated the cache first and already has the file
        # mapped, which makes it unreplaceable on Windows. Its copy came from
        # the same source, so the loser of the race is done either way.
        if not os.path.exists(target):
            raise
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _ladybug_vendors_openssl() -> bool:
    """Whether the installed ladybug wheel carries its own OpenSSL.

    Wheels through 0.18.x vendored it (under delvewheel's hashed names) and a
    fixed release will vendor it again. In both cases ladybug's own
    ``ladybug.libs`` directory is already registered by the delvewheel patch in
    ``ladybug/__init__.py``, and this shim has to stay out of the way.
    """
    spec = importlib.util.find_spec("ladybug")
    if spec is None or not spec.origin:
        return False

    package_directory = os.path.dirname(spec.origin)
    libs_directory = os.path.join(os.path.dirname(package_directory), "ladybug.libs")
    try:
        vendored = os.listdir(libs_directory)
    except OSError:
        return False
    return any(name.startswith("libssl-3-x64") for name in vendored)

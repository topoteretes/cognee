"""Fail when an installed package bundles a program under copyleft terms.

A package's declared license describes its Python code, not every file in its
wheel. ``pypandoc-binary`` declares MIT and installs the ``pandoc`` executable,
whose terms are GPL, with no pandoc license file beside it. Metadata-based
license checks report MIT and pass.

This reads the installed environment instead. For every file an installed
distribution records, it keeps the native binaries (ELF, Mach-O or PE) and
scans their bytes for the copyleft notices a program embeds. The executable
bit is not required: some packages ``chmod`` their bundled tool at first use.
A package's own extension modules and shared libraries are skipped, since
they are built from its source. Libraries that auditwheel, delocate or
delvewheel vendor into ``<pkg>.libs/`` or ``.dylibs/`` are third-party code
and are scanned.

Run it with the interpreter of the environment to check:

    uv run --no-sync python scripts/check_bundled_program_licenses.py
"""

import re
import sys
from importlib.metadata import distributions
from pathlib import Path

# Notices a program embeds when it carries copyleft terms.
COPYLEFT = {
    b"GNU AFFERO GENERAL PUBLIC LICENSE": "AGPL",
    b"GNU LESSER GENERAL PUBLIC LICENSE": "LGPL",
    b"GNU LIBRARY GENERAL PUBLIC LICENSE": "LGPL",  # LGPL v2.0 title
    b"GNU GENERAL PUBLIC LICENSE": "GPL",
}

# Magic numbers of native executables: ELF, Mach-O (thin and fat), PE.
NATIVE = (
    b"\x7fELF",
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xfe\xed\xfa\xce",
    b"\xca\xfe\xba\xbe",
    b"MZ",
)

# Built from the package's own source; not a vendored program.
LIBRARY_SUFFIXES = (".so", ".pyd", ".dylib", ".dll")

# Distributions a maintainer has reviewed and accepted, with the reason.
# Keyed by canonical name ("pypandoc-binary"). Remove an entry when the
# package is dropped.
REVIEWED: dict[str, str] = {}

CHUNK = 1 << 20


def is_vendored_library_dir(name: str) -> bool:
    """Directories where wheel-repair tools copy third-party libraries."""
    return name.endswith(".libs") or name == ".dylibs"


def is_native_program(path: Path) -> bool:
    """Keep native binaries, except the package's own libraries."""
    if not path.is_file() or path.is_symlink():
        return False
    is_library = path.name.endswith(LIBRARY_SUFFIXES) or ".so." in path.name
    if is_library and not is_vendored_library_dir(path.parent.name):
        return False
    with path.open("rb") as handle:
        return handle.read(4).startswith(NATIVE)


def embedded_copyleft(path: Path) -> set[str]:
    """Return the copyleft licenses whose notice appears in the file's bytes."""
    found, tail = set(), b""
    overlap = max(len(phrase) for phrase in COPYLEFT)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            window = (tail + chunk).upper()
            found |= {name for phrase, name in COPYLEFT.items() if phrase in window}
            tail = chunk[-overlap:]
    return found


def canonical(name: str) -> str:
    """Normalize a distribution name the way PyPI does (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    scanned, flagged = 0, []
    for dist in distributions():
        name = canonical(dist.metadata["Name"])
        for record in dist.files or []:
            path = Path(dist.locate_file(record))
            if not is_native_program(path):
                continue
            scanned += 1
            licenses = embedded_copyleft(path)
            if licenses:
                flagged.append((name, record, sorted(licenses)))

    print(f"Scanned {scanned} bundled native programs.", flush=True)
    failures = 0
    for name, record, licenses in sorted(flagged):
        if name in REVIEWED:
            print(f"REVIEWED {name}: {record} ({', '.join(licenses)}): {REVIEWED[name]}")
            continue
        failures += 1
        print(f"COPYLEFT {name}: {record} embeds {', '.join(licenses)}", file=sys.stderr)
    if failures:
        print(
            f"{failures} bundled program(s) carry copyleft terms their package does not "
            "declare. Drop the dependency, or record a reviewed decision in REVIEWED.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

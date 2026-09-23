"""Fail when an installed package's license would extend to cognee's own code.

cognee is Apache-2.0. It imports its dependencies and loads the native
libraries they vendor into its own process. When that code is under strong
copyleft (GPL or AGPL), distributing cognee with it puts the combined work
under those terms. That is the only case this checks, in two places per
installed distribution:

- Its declared license, for the code cognee imports: ``License-Expression``,
  the ``License`` field when it is a name rather than a full text, and the
  ``License ::`` classifiers. Each is checked on its own, since they can
  disagree (``pi-heif`` declares BSD-3-Clause next to an LGPLv3 classifier).
- The native libraries it vendors into ``<pkg>.libs/`` or ``.dylibs/``
  (auditwheel, delocate, delvewheel): the ``License:`` lines of its bundled
  license manifests, and the license notices embedded in the libraries.

Not flagged, because they leave cognee's license unchanged:

- Weak copyleft (LGPL, MPL): it covers the library itself, not code that
  imports or links it.
- Separate programs a package ships and runs as a subprocess, such as
  ``pypandoc-binary``'s GPL ``pandoc``: that is aggregation, not a combined
  work. Only libraries are scanned.
- GPL with a linking exception (``GPL-3.0-or-later WITH GCC-exception-3.1``,
  the libgfortran runtime numpy and scipy vendor) or with a permissive ``OR``
  alternative (matplotlib's FreeType, ``FTL OR GPL-2.0-or-later``).
- Proprietary terms (NVIDIA's CUDA wheels): they restrict those files, not
  cognee's license.

Full license texts are not searched: they mention the GPL in passing
("GPL-compatible", MPL's "Secondary License") far more often than they grant it.

Run it with the interpreter of the environment to check:

    uv run --no-sync python scripts/check_license_compatibility.py
"""

import re
import sys
from importlib.metadata import Distribution, distributions
from pathlib import Path

# License titles, as classifiers spell them and native libraries embed them.
TITLES = {
    "GNU AFFERO GENERAL PUBLIC LICENSE": "AGPL",
    "GNU LESSER GENERAL PUBLIC LICENSE": "LGPL",
    "GNU LIBRARY GENERAL PUBLIC LICENSE": "LGPL",  # LGPL v2.0 title
    "GNU LIBRARY OR LESSER GENERAL PUBLIC LICENSE": "LGPL",  # classifier wording
    "GNU GENERAL PUBLIC LICENSE": "GPL",
}

# GNU identifiers as SPDX ids, classifiers and manifests spell them
# ("GPL-2.0-only", "(AGPLv3)", "GPLv2+"), but not "GPL-compatible".
GNU_ID = re.compile(r"(?<![A-Z])(AGPL|LGPL|GPL)(?=V?\d|-\d|[\s)\],;+.]|$)")

# The licenses whose terms extend to code that imports or links them.
STRONG_COPYLEFT = {"GPL", "AGPL"}

# Exceptions that let code under another license link the library.
LINKING_EXCEPTION = re.compile(r"\bWITH\s+(GCC|LLVM|CLASSPATH)-EXCEPTION")

# A license line inside a bundled license manifest ("License: GPL-2.0-only").
MANIFEST_LICENSE_LINE = re.compile(r"^\s*LICENSE:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

# A License field longer than this is license text, not a license name.
MAX_LICENSE_NAME = 100

# Magic numbers of native binaries: ELF, Mach-O (thin and fat), PE.
NATIVE = (
    b"\x7fELF",
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xfe\xed\xfa\xce",
    b"\xca\xfe\xba\xbe",
    b"MZ",
)

# Distributions a maintainer has reviewed and accepted, with the reason.
# Keyed by canonical name ("some-package"). Remove an entry when the
# package is dropped.
REVIEWED: dict[str, str] = {}

CHUNK = 1 << 20


def strong_copyleft(alternatives: list[str], expressions: bool = True) -> set[str]:
    """Strong copyleft licenses that apply when any one of ``alternatives`` may be chosen.

    Empty when some alternative is free of strong copyleft or carries a
    linking exception. With ``expressions``, each alternative may itself be an
    ``OR`` expression; classifiers are prose ("GNU Library or Lesser ...") and
    are not split.
    """
    required: set[str] = set()
    for expression in alternatives:
        parts = re.split(r"\s+OR\s+", expression.upper()) if expressions else [expression.upper()]
        for alternative in parts:
            if LINKING_EXCEPTION.search(alternative):
                return set()
            terms = set(GNU_ID.findall(alternative))
            terms |= {name for title, name in TITLES.items() if title in alternative}
            if not terms & STRONG_COPYLEFT:
                return set()
            required |= terms & STRONG_COPYLEFT
    return required


def is_vendored_library(path: Path) -> bool:
    """A native library that a wheel-repair tool copied into the wheel."""
    if not (path.parent.name.endswith(".libs") or path.parent.name == ".dylibs"):
        return False
    if not path.is_file() or path.is_symlink():
        return False
    with path.open("rb") as handle:
        return handle.read(4).startswith(NATIVE)


def embedded_strong_copyleft(path: Path) -> set[str]:
    """Strong copyleft licenses whose title a native library embeds.

    LGPL texts quote the GPL's title ("incorporates ... the GNU General Public
    License"), so a library that also embeds an LGPL title is weak copyleft.
    """
    found, tail = set(), b""
    titles = {title.encode(): name for title, name in TITLES.items()}
    overlap = max(len(title) for title in titles)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            window = (tail + chunk).upper()
            found |= {name for title, name in titles.items() if title in window}
            tail = chunk[-overlap:]
    return set() if "LGPL" in found else found & STRONG_COPYLEFT


def declared_licenses(dist: Distribution) -> list[tuple[str, list[str], bool]]:
    """The distribution's declared license sources: (where, alternatives, expressions)."""
    metadata = dist.metadata
    license_field = (metadata.get("License") or "").strip()
    if "\n" in license_field or len(license_field) > MAX_LICENSE_NAME:
        license_field = ""
    classifiers = [
        c.split(" :: ", 1)[1]
        for c in metadata.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    return [
        ("License-Expression", [metadata.get("License-Expression") or ""], True),
        ("License", [license_field], True),
        ("classifiers", classifiers, False),
    ]


def manifest_lines(dist: Distribution) -> list[tuple[str, str]]:
    """``License:`` lines of the license files in the distribution's metadata."""
    lines = []
    for record in dist.files or []:
        path = str(record)
        name = path.rsplit("/", 1)[-1].upper()
        if ".dist-info/" in path and any(word in name for word in ("LICEN", "COPYING")):
            text = Path(dist.locate_file(record)).read_bytes().decode("utf-8", "ignore")
            where = path.split(".dist-info/", 1)[1]
            lines += [(where, line.strip()) for line in MANIFEST_LICENSE_LINE.findall(text)]
    return lines


def findings(dist: Distribution) -> list[tuple[str, str]]:
    """Strong copyleft the distribution declares, lists or vendors: (where, what)."""
    found = []
    for where, alternatives, expressions in declared_licenses(dist):
        if alternatives and all(alternatives):
            terms = strong_copyleft(alternatives, expressions)
            if terms:
                found.append((where, f"declares {', '.join(sorted(terms))}"))
    for where, line in manifest_lines(dist):
        terms = strong_copyleft([line])
        if terms:
            found.append((where, f"lists {', '.join(sorted(terms))} ({line[:60]})"))
    for record in dist.files or []:
        path = Path(dist.locate_file(record))
        if is_vendored_library(path):
            terms = embedded_strong_copyleft(path)
            if terms:
                found.append((str(record), f"embeds {', '.join(sorted(terms))}"))
    return found


def canonical(name: str) -> str:
    """Normalize a distribution name the way PyPI does (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    checked, flagged = 0, set()
    for dist in distributions():
        checked += 1
        name = canonical(dist.metadata["Name"])
        flagged |= {(name, where, what) for where, what in findings(dist)}

    print(f"Checked {checked} installed distributions.", flush=True)
    failures = 0
    for name, where, what in sorted(flagged):
        if name in REVIEWED:
            print(f"REVIEWED {name}: {where} {what}: {REVIEWED[name]}")
            continue
        failures += 1
        print(f"COPYLEFT {name}: {where} {what}", file=sys.stderr)
    if failures:
        print(
            f"{failures} finding(s) of strong copyleft (GPL, AGPL) that would extend to "
            "cognee's Apache-2.0 code. Drop the dependency, or record a reviewed decision "
            "in REVIEWED.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

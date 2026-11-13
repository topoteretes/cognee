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
  Most libraries embed no notice, so this is best-effort: the vendored
  libraries of a distribution with no manifest line, and no notice of their
  own, are listed in the output instead of passing silently.

Not flagged, because they leave cognee's license unchanged:

- Weak copyleft (LGPL, MPL): it covers the library itself, not code that
  imports or links it.
- Separate programs a package ships and runs as a subprocess, such as
  ``pypandoc-binary``'s GPL ``pandoc``: that is aggregation, not a combined
  work. Only libraries are scanned.
- GPL with a linking exception (``GPL-3.0-or-later WITH GCC-exception-3.1``,
  the libgfortran runtime numpy and scipy vendor) or with a recognised
  permissive or weak-copyleft ``OR`` alternative (matplotlib's FreeType,
  ``FTL OR GPL-2.0-or-later``). An alternative that is not a recognised open
  license -- a commercial license, ``LicenseRef-*``, anything unknown -- does
  not excuse the copyleft: it is not a choice an Apache-2.0 distribution can
  make (PyMuPDF, ``GNU AFFERO GPL 3.0 or Artifex Commercial License``).
- Proprietary terms (NVIDIA's CUDA wheels): they restrict those files, not
  cognee's license.

Full license texts are not searched: they mention the GPL in passing
("GPL-compatible", MPL's "Secondary License") far more often than they grant it.
Only their first line, the license's title, is read when a ``License`` field
holds the full text.

The result covers the one environment it runs in: ``uv.lock`` resolves other
packages and versions on other Python versions and platforms.

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
    "GNU AFFERO GPL": "AGPL",  # PyMuPDF's License field
    "GNU LESSER GENERAL PUBLIC LICENSE": "LGPL",
    "GNU LIBRARY GENERAL PUBLIC LICENSE": "LGPL",  # LGPL v2.0 title
    "GNU LIBRARY OR LESSER GENERAL PUBLIC LICENSE": "LGPL",  # classifier wording
    "GNU GENERAL PUBLIC LICENSE": "GPL",
}

# GNU identifiers as SPDX ids, classifiers and manifests spell them
# ("GPL-2.0-only", "(AGPLv3)", "GPLv2+", "GPL/Commercial"), but not
# "GPL-compatible", and not the "GPL" inside "AFFERO GPL", which TITLES
# reports as AGPL.
GNU_ID = re.compile(r"(?<![A-Z])(?<!AFFERO )(AGPL|LGPL|GPL)(?=V?\d|-\d|[\s)\],;+./]|$)")

# The licenses whose terms extend to code that imports or links them.
STRONG_COPYLEFT = {"GPL", "AGPL"}

# Exceptions that let code under another license link the library, as SPDX
# expressions ("GPL-3.0-or-later WITH GCC-exception-3.1") and the deprecated
# SPDX ids numpy 2.2 still lists ("GPL-3.0-with-GCC-exception") spell them.
LINKING_EXCEPTION = re.compile(r"(\bWITH\s+|-WITH-)(GCC|LLVM|CLASSPATH)-EXCEPTION")

# Open licenses an Apache-2.0 distribution can choose when a package offers
# them as an ``OR`` alternative to strong copyleft: permissive and weak
# copyleft, as SPDX ids and as classifiers and License fields spell them.
# Anything else offered as the alternative (a commercial license,
# ``LicenseRef-*``, free text) is not a choice this distribution can make.
OPEN_ALTERNATIVE = re.compile(
    r"\b(MIT|BSD|APACHE|ISC|MPL|MOZILLA|LGPL|PSF|PYTHON SOFTWARE FOUNDATION|ZLIB|UNLICENSE"
    r"|CC0|BSL|BOOST|0BSD|FTL|FREETYPE|HPND|EPL|ECLIPSE|CDDL|ARTISTIC|WTFPL|OFL|SIL OPEN FONT"
    r"|ZPL|ZOPE|PUBLIC DOMAIN)\b"
)

# A license line inside a bundled license manifest ("License: GPL-2.0-only").
MANIFEST_LICENSE_LINE = re.compile(r"^\s*LICENSE:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

# The brackets and operators of a license expression. An operator needs
# whitespace on both sides, so "GPL-2.0-or-later" stays one license.
EXPRESSION_TOKEN = re.compile(r"(\(|\)|\s+(?:AND|OR)\s+)")

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


def strong_copyleft(alternatives: list[str]) -> set[str]:
    """Strong copyleft licenses that apply when any one of ``alternatives`` may be chosen.

    Empty when some alternative carries a linking exception or is a recognised
    open license (``OPEN_ALTERNATIVE``) free of strong copyleft. An alternative
    that is neither -- a commercial license, ``LicenseRef-*``, unknown text --
    excuses nothing. Each alternative may itself be an expression: brackets
    bind first, then ``AND``, then ``OR``, so in ``(MIT OR Apache-2.0) AND
    GPL-3.0-only`` the GPL applies whichever bracketed license is chosen.
    """
    results = [evaluate(expression.upper()) for expression in alternatives]
    if any(is_open for _, is_open in results):
        return set()
    return set().union(*(required for required, _ in results))


def evaluate(expression: str) -> tuple[set[str], bool]:
    """One license expression: (strong copyleft that applies, whether it is open).

    Open means an Apache-2.0 distribution can take it as an alternative to
    strong copyleft. Unmatched closing brackets are ignored.
    """
    tokens, depth = [], 0
    for token in EXPRESSION_TOKEN.split(expression):
        token = token.strip()
        if token == ")":
            if not depth:
                continue
            depth -= 1
        elif token == "(":
            depth += 1
        if token:
            tokens.append(token)
    result, _ = parse_or(tokens, 0)
    return result


def parse_or(tokens: list[str], i: int) -> tuple[tuple[set[str], bool], int]:
    """``OR`` from ``tokens[i]``: an open branch excuses the others, else all apply."""
    branches = []
    while True:
        branch, i = parse_and(tokens, i)
        branches.append(branch)
        if i < len(tokens) and tokens[i] == "OR":
            i += 1
        else:
            break
    if any(is_open for _, is_open in branches):
        return (set(), True), i
    return (set().union(*(required for required, _ in branches)), False), i


def parse_and(tokens: list[str], i: int) -> tuple[tuple[set[str], bool], int]:
    """``AND`` from ``tokens[i]``, and terms written side by side ("GPL v3 (GPLv3)"):
    every term applies, and the whole is open only when every term is."""
    terms = []
    while i < len(tokens) and tokens[i] not in ("OR", ")"):
        if tokens[i] == "AND":
            i += 1
        elif tokens[i] == "(":
            term, i = parse_or(tokens, i + 1)
            if i < len(tokens):
                i += 1  # the closing bracket
            terms.append(term)
        else:
            terms.append(license_term(tokens[i]))
            i += 1
    required = set().union(*(required for required, _ in terms))
    return (required, bool(terms) and all(is_open for _, is_open in terms)), i


def license_term(text: str) -> tuple[set[str], bool]:
    """A single license, without brackets or operators: (strong copyleft, whether open)."""
    if LINKING_EXCEPTION.search(text):
        return set(), True
    names = set(GNU_ID.findall(text))
    names |= {name for title, name in TITLES.items() if title in text}
    if names & STRONG_COPYLEFT:
        return names & STRONG_COPYLEFT, False
    return set(), bool(OPEN_ALTERNATIVE.search(text))


def is_vendored_library(path: Path) -> bool:
    """A native library that a wheel-repair tool copied into the wheel."""
    if not (path.parent.name.endswith(".libs") or path.parent.name == ".dylibs"):
        return False
    if not path.is_file() or path.is_symlink():
        return False
    with path.open("rb") as handle:
        return handle.read(4).startswith(NATIVE)


def embedded_licenses(path: Path) -> set[str]:
    """The licenses whose title a native library embeds."""
    found, tail = set(), b""
    titles = {title.encode(): name for title, name in TITLES.items()}
    overlap = max(len(title) for title in titles)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            window = (tail + chunk).upper()
            found |= {name for title, name in titles.items() if title in window}
            tail = chunk[-overlap:]
    return found


def vendored_libraries(dist: Distribution) -> list[tuple[str, set[str]]]:
    """The native libraries the distribution vendors: (where, embedded licenses)."""
    libraries = []
    for record in dist.files or []:
        path = Path(dist.locate_file(record))
        if is_vendored_library(path):
            libraries.append((str(record), embedded_licenses(path)))
    return libraries


def declared_licenses(dist: Distribution) -> list[tuple[str, list[str]]]:
    """The distribution's declared license sources: (where, alternatives)."""
    metadata = dist.metadata
    license_field = (metadata.get("License") or "").strip()
    if "\n" in license_field:
        # License text, not a name. A text starts with its own title, and only
        # its later lines mention the GPL in passing.
        license_field = license_field.split("\n", 1)[0].strip()
    classifiers = [
        c.split(" :: ", 1)[1]
        for c in metadata.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    return [
        ("License-Expression", [metadata.get("License-Expression") or ""]),
        ("License", [license_field]),
        ("classifiers", classifiers),
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


def findings(
    dist: Distribution, libraries: list[tuple[str, set[str]]] | None = None
) -> list[tuple[str, str]]:
    """Strong copyleft the distribution declares, lists or vendors: (where, what).

    ``libraries`` is ``vendored_libraries(dist)`` when the caller already has it.
    """
    found = []
    for where, alternatives in declared_licenses(dist):
        if alternatives and all(alternatives):
            terms = strong_copyleft(alternatives)
            if terms:
                found.append((where, f"declares {', '.join(sorted(terms))}"))
    for where, line in manifest_lines(dist):
        terms = strong_copyleft([line])
        if terms:
            found.append((where, f"lists {', '.join(sorted(terms))} ({line[:60]})"))
    for where, embedded in vendored_libraries(dist) if libraries is None else libraries:
        # LGPL texts quote the GPL's title ("incorporates ... the GNU General
        # Public License"), so a library that also embeds an LGPL title is weak
        # copyleft.
        terms = set() if "LGPL" in embedded else embedded & STRONG_COPYLEFT
        if terms:
            found.append((where, f"embeds {', '.join(sorted(terms))}"))
    return found


def canonical(name: str) -> str:
    """Normalize a distribution name the way PyPI does (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def has_license_metadata(dist: Distribution) -> bool:
    """Whether the distribution declares a license name anywhere (for a License
    field holding full license text, its first line, see ``declared_licenses``)."""
    return any(alternatives and all(alternatives) for _, alternatives in declared_licenses(dist))


def main() -> int:
    checked, flagged, undeclared, unlabelled = 0, set(), [], []
    for dist in distributions():
        checked += 1
        name = canonical(dist.metadata["Name"])
        libraries = vendored_libraries(dist)
        flagged |= {(name, where, what) for where, what in findings(dist, libraries)}
        if not has_license_metadata(dist):
            undeclared.append(name)
        if not manifest_lines(dist):
            unlabelled += [where for where, embedded in libraries if not embedded]

    print(f"Checked {checked} installed distributions.", flush=True)
    # Neither list is a failure: there is nothing to check against. They are
    # printed so the blind spots are visible in the job output instead of silent.
    if undeclared:
        print(f"No license name declared ({len(undeclared)}): {', '.join(sorted(undeclared))}")
    if unlabelled:
        print(f"Vendored libraries with no license manifest or notice ({len(unlabelled)}):")
        for where in sorted(unlabelled):
            print(f"  {where}")
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

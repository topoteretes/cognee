"""Fail when an installed package carries copyleft, proprietary or no license terms.

A package's declared license describes its Python code, not every file in its
wheel. ``pypandoc-binary`` declares MIT and installs the ``pandoc`` executable,
whose terms are GPL, with no pandoc license file beside it. ``pi-heif``
declares BSD-3-Clause and vendors LGPLv3 ``libheif``, saying so only in a
classifier and a bundled license manifest. A check that reads the ``License``
field alone reports both as permissive and passes.

This reads the installed environment instead, three ways per distribution:

- Native binaries (ELF, Mach-O or PE) it records are scanned for the copyleft
  notices a program embeds. The executable bit is not required: some packages
  ``chmod`` their bundled tool at first use. A package's own extension modules
  and shared libraries are skipped, since they are built from its source.
  Libraries that auditwheel, delocate or delvewheel vendor into
  ``<pkg>.libs/`` or ``.dylibs/`` are third-party code and are scanned.
- Declared terms: ``License-Expression``, a short ``License`` field, the
  ``License ::`` classifiers, and ``License:`` lines in bundled license files
  (the manifests wheels use to list vendored libraries). Full license texts
  are not searched: they mention the GPL in passing ("GPL-compatible", MPL's
  "Secondary License") far more often than they grant it.
- Proprietary terms, or no license information at all.

A copyleft term does not count when an ``OR`` alternative is permissive, or
when it carries a linking exception (``GPL-3.0-or-later WITH
GCC-exception-3.1``, the libgfortran runtime numpy and scipy vendor).

Run it with the interpreter of the environment to check:

    uv run --no-sync python scripts/check_bundled_program_licenses.py
"""

import re
import sys
from importlib.metadata import Distribution, distributions
from pathlib import Path

# Notices a program embeds when it carries copyleft terms.
COPYLEFT = {
    b"GNU AFFERO GENERAL PUBLIC LICENSE": "AGPL",
    b"GNU LESSER GENERAL PUBLIC LICENSE": "LGPL",
    b"GNU LIBRARY GENERAL PUBLIC LICENSE": "LGPL",  # LGPL v2.0 title
    b"GNU GENERAL PUBLIC LICENSE": "GPL",
}

# Copyleft identifiers as SPDX ids, classifiers and manifests spell them
# ("LGPL-2.1-only", "(LGPLv3)", "GPLv2+"), but not "GPL-compatible".
COPYLEFT_ID = re.compile(r"(?<![A-Z])(AGPL|LGPL|GPL)(?=V?\d|-\d|[\s)\],;+.]|$)")

# Exceptions that let non-copyleft code link against the licensed library.
LINKING_EXCEPTION = re.compile(r"\bWITH\s+(GCC|LLVM|CLASSPATH)-EXCEPTION")

# A license line inside a bundled license manifest ("License: LGPLv3").
MANIFEST_LICENSE_LINE = re.compile(r"^\s*LICENSE:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

# A License field longer than this is license text, not a license name.
MAX_LICENSE_NAME = 100

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


def copyleft_terms(alternatives: list[str], expressions: bool = True) -> set[str]:
    """Copyleft licenses that apply when any one of ``alternatives`` may be chosen.

    Empty when some alternative is free of copyleft or carries a linking
    exception. With ``expressions``, each alternative may itself be an ``OR``
    expression; classifiers are prose ("GNU Library or Lesser ...") and are not
    split.
    """
    required: set[str] = set()
    for expression in alternatives:
        parts = re.split(r"\s+OR\s+", expression.upper()) if expressions else [expression.upper()]
        for alternative in parts:
            if LINKING_EXCEPTION.search(alternative):
                return set()
            terms = set(COPYLEFT_ID.findall(alternative))
            terms |= {name for phrase, name in COPYLEFT.items() if phrase.decode() in alternative}
            if not terms:
                return set()
            required |= terms
    return required


def license_files(dist: Distribution) -> list[tuple[str, str]]:
    """The license, copying and notice files in the distribution's metadata."""
    found = []
    for record in dist.files or []:
        path = str(record)
        if ".dist-info/" not in path:
            continue
        name = path.rsplit("/", 1)[-1].upper()
        if any(word in name for word in ("LICEN", "COPYING", "NOTICE")):
            text = Path(dist.locate_file(record)).read_bytes().decode("utf-8", "ignore")
            found.append((path.split(".dist-info/", 1)[1], text))
    return found


def declared_findings(dist: Distribution) -> list[tuple[str, str]]:
    """Copyleft, proprietary or missing terms the distribution's metadata declares."""
    metadata = dist.metadata
    expression = metadata.get("License-Expression") or ""
    license_field = (metadata.get("License") or "").strip()
    if "\n" in license_field or len(license_field) > MAX_LICENSE_NAME:
        license_field = ""
    classifiers = [
        c.split(" :: ", 1)[1]
        for c in metadata.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    files = license_files(dist)

    findings = []
    # Each source on its own: pi-heif's BSD field must not excuse its LGPL classifier.
    for where, alternatives, expressions in (
        ("License-Expression", [expression], True),
        ("License", [license_field], True),
        ("classifiers", classifiers, False),
    ):
        if alternatives and all(alternatives):
            terms = copyleft_terms(alternatives, expressions)
            if terms:
                findings.append((where, f"declares {', '.join(sorted(terms))}"))
    for path, text in files:
        for line in MANIFEST_LICENSE_LINE.findall(text):
            terms = copyleft_terms([line])
            if terms:
                findings.append((path, f"lists {', '.join(sorted(terms))} ({line.strip()[:60]})"))

    declared = [value for value in (expression, license_field, *classifiers) if value]
    if declared and all("PROPRIETARY" in value.upper() for value in declared):
        findings.append(("metadata", f"declares only proprietary terms ({declared[0]})"))
    if not declared and not files:
        findings.append(("metadata", "declares no license"))
    return findings


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
                flagged.append((name, str(record), f"embeds {', '.join(sorted(licenses))}"))
        flagged += [(name, where, what) for where, what in declared_findings(dist)]

    print(f"Scanned {scanned} bundled native programs.", flush=True)
    failures = 0
    for name, where, what in sorted(set(flagged)):
        if name in REVIEWED:
            print(f"REVIEWED {name}: {where} {what}: {REVIEWED[name]}")
            continue
        failures += 1
        print(f"LICENSE {name}: {where} {what}", file=sys.stderr)
    if failures:
        print(
            f"{failures} finding(s) of copyleft, proprietary or undeclared license terms. "
            "Drop the dependency, or record a reviewed decision in REVIEWED.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

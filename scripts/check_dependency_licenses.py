# /// script
# requires-python = ">=3.12"
# dependencies = ["license-expression==30.4.4"]
# ///
"""Audit the license of every distribution artifact this repository installs.

A registry record describes a published package, not what its archive contains.
``pypandoc-binary`` declares MIT and ships the ``pandoc`` executable, whose own
terms are GPL and whose license text the wheel does not carry. Reading metadata
alone cannot see that, and a gate that only diffs lockfiles never re-examines a
dependency once it is locked.

So this audit reads the artifacts. Wheels are read in place over HTTP range
requests, which keeps a 25 MB wheel down to a few kilobytes and never executes
packaging or build hooks. Each artifact is then judged three ways:

1. the license its own metadata declares,
2. the license texts it bundles, scanned for copyleft and source-available
   terms that a permissive declaration would not cover,
3. the third-party programs it carries. Those are the payload registry metadata
   is least likely to describe, so their bytes are scanned for the license
   notices the program embeds.

Anything unresolved is reported and fails the audit until a maintainer records a
decision in the policy file.
"""

import argparse
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import tomllib
from license_expression import ExpressionError, get_spdx_licensing

LICENSING = get_spdx_licensing()

NPM_REGISTRY = "https://registry.npmjs.org/"
# Some indexes refuse the default Python agent: the PyTorch index answers
# 403 to "Python-urllib" and 206 to a named one.
USER_AGENT = "cognee-license-audit/1.0 (+https://github.com/topoteretes/cognee)"
FIRST_PARTY = {"cognee", "cognee-mcp"}

# Archive members whose text is read and classified.
LICENSE_NAMES = re.compile(r"(^|/)(LICEN[CS]E|COPYING|COPYRIGHT|NOTICE)", re.IGNORECASE)

# Executable payloads that are neither Python extension modules (.so/.pyd) nor
# console-script shims. An empty suffix is the interesting case: a vendored
# native program, which is how pandoc reaches disk.
PROGRAM_SUFFIXES = {"", ".exe", ".bin", ".run", ".appimage"}
PROGRAM_EXEMPT = re.compile(r"\.(dist-info|data/scripts|egg-info)/", re.IGNORECASE)

# Phrases identifying terms the allowlist does not grant. Matched against
# bundled license text and program bytes, so a permissive declaration cannot
# hide a copyleft component vendored beside it.
RESTRICTED_TEXTS = (
    ("GNU AFFERO GENERAL PUBLIC LICENSE", "AGPL-3.0-only"),
    ("GNU LESSER GENERAL PUBLIC LICENSE", "LGPL-3.0-only"),
    ("GNU LIBRARY GENERAL PUBLIC LICENSE", "LGPL-2.0-only"),
    ("GNU GENERAL PUBLIC LICENSE", "GPL-3.0-only"),
    ("MOZILLA PUBLIC LICENSE", "MPL-2.0"),
    ("COMMON DEVELOPMENT AND DISTRIBUTION LICENSE", "CDDL-1.0"),
    ("ECLIPSE PUBLIC LICENSE", "EPL-2.0"),
    ("BUSINESS SOURCE LICENSE", "BUSL-1.1"),
    ("SERVER SIDE PUBLIC LICENSE", "SSPL-1.0"),
    ("COMMONS CLAUSE", "LicenseRef-Commons-Clause"),
    ("POLYFORM NONCOMMERCIAL", "LicenseRef-PolyForm-Noncommercial"),
    ("POLYFORM SHIELD", "LicenseRef-PolyForm-Shield"),
    ("ATTRIBUTION-NONCOMMERCIAL", "LicenseRef-NonCommercial"),
)

# Free-text values the pre-PEP-639 "License:" field still carries. Mapping them
# is what keeps the audit's output about real license questions instead of
# spelling. Each entry must name the same license the text does, never a more
# permissive one; anything ambiguous maps to an expression listing the options.
LEGACY_LICENSES = {
    "MIT": "MIT",
    "MIT LICENSE": "MIT",
    "THE MIT LICENSE": "MIT",
    "MIT LICENCE": "MIT",
    "APACHE": "Apache-2.0",
    "APACHE 2": "Apache-2.0",
    "APACHE 2.0": "Apache-2.0",
    "APACHE LICENSE 2.0": "Apache-2.0",
    "APACHE LICENSE, VERSION 2.0": "Apache-2.0",
    "APACHE SOFTWARE LICENSE": "Apache-2.0",
    "APACHE SOFTWARE LICENSE 2.0": "Apache-2.0",
    "ASL 2": "Apache-2.0",
    "BSD": "BSD-2-Clause OR BSD-3-Clause",
    "BSD LICENSE": "BSD-2-Clause OR BSD-3-Clause",
    "BSD-3-CLAUSE": "BSD-3-Clause",
    "BSD 3-CLAUSE": "BSD-3-Clause",
    "BSD 3-CLAUSE LICENSE": "BSD-3-Clause",
    "3-CLAUSE BSD LICENSE": "BSD-3-Clause",
    "NEW BSD LICENSE": "BSD-3-Clause",
    "MODIFIED BSD LICENSE": "BSD-3-Clause",
    "BSD-2-CLAUSE": "BSD-2-Clause",
    "BSD 2-CLAUSE": "BSD-2-Clause",
    "SIMPLIFIED BSD LICENSE": "BSD-2-Clause",
    "ISC": "ISC",
    "ISC LICENSE": "ISC",
    "MPL 2.0": "MPL-2.0",
    "MOZILLA PUBLIC LICENSE 2.0": "MPL-2.0",
    "THE UNLICENSE": "Unlicense",
    "PSF": "PSF-2.0",
    "PYTHON SOFTWARE FOUNDATION LICENSE": "PSF-2.0",
}

# What setuptools writes when a project declares nothing at all.
UNDECLARED = {"UNKNOWN", "NONE", "OTHER/PROPRIETARY LICENSE", "DUAL LICENSE"}

MAX_TEXT_BYTES = 256 * 1024
SCAN_CHUNK = 1 << 20
# A tarball larger than this is written to disk instead of held in memory.
SPOOL_BYTES = 8 << 20


@dataclass(frozen=True, order=True)
class Artifact:
    """One downloadable distribution file named by a lockfile."""

    ecosystem: str
    name: str
    version: str
    kind: str
    url: str
    size: int = 0

    @property
    def key(self) -> str:
        return f"{self.ecosystem}:{self.name}@{self.version}"

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass
class Inspection:
    """What reading one artifact revealed."""

    declared: str = ""
    bundled: dict = field(default_factory=dict)
    programs: dict = field(default_factory=dict)


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only view of a remote file, served by range requests.

    Handing this to ``zipfile`` reads a wheel's index and a few small members
    without transferring the archive, which is what makes auditing every locked
    wheel affordable.
    """

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self.size - self.pos)
        if count <= 0:
            return 0
        data = fetch(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + count - 1}"})
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def attempt(url: str, headers: dict, method: str, sink=None, attempts: int = 4):
    """Perform one request, retrying only transient transport failures.

    The retry has to cover the response body too. Reading a large artifact times
    out far more often than opening the connection does. With a ``sink`` the
    body is streamed into it rather than returned, which is what keeps a 75 MB
    source distribution off the heap.
    """
    sent = {"User-Agent": USER_AGENT, **headers}
    for number in range(attempts):
        try:
            with urlopen(Request(url, headers=sent, method=method), timeout=120) as response:
                if method == "HEAD":
                    return dict(response.headers)
                if sink is None:
                    return response.read()
                sink.seek(0)
                sink.truncate()
                shutil.copyfileobj(response, sink, SCAN_CHUNK)
                sink.seek(0)
                return sink
        except HTTPError as error:
            if error.code not in (408, 425, 429, 500, 502, 503, 504) or number == attempts - 1:
                raise
        except (URLError, TimeoutError, ConnectionError, OSError):
            if number == attempts - 1:
                raise
        time.sleep(2**number)
    raise OSError(f"unreachable: {url}")


def fetch(url: str, headers: dict | None = None) -> bytes:
    return attempt(url, headers or {}, "GET")


def download(url: str):
    """Stream an artifact into a file that spills to disk once it grows.

    A tarball cannot be read in place, and holding several of them in memory at
    once is what the concurrency makes expensive.
    """
    return attempt(url, {}, "GET", sink=tempfile.SpooledTemporaryFile(max_size=SPOOL_BYTES))


def remote_size(url: str) -> int:
    """Ask the host how large an artifact is, for locks that do not record it."""
    length = attempt(url, {}, "HEAD").get("Content-Length")
    if not length:
        raise ValueError(f"host reports no size for {url}")
    return int(length)


def artifacts_from_lock(path: str, content: str) -> set:
    """List every artifact a lockfile can install, first-party roots excluded."""
    found = set()
    if Path(path).name == "uv.lock":
        for item in tomllib.loads(content)["package"]:
            source, name = item["source"], item["name"]
            if source.keys() & {"editable", "virtual", "directory"} and name in FIRST_PARTY:
                continue
            if list(source) != ["registry"]:
                raise ValueError(f"{name}: source {source} requires manual review")
            # uv installs a wheel whenever one matches the interpreter and
            # platform; an sdist is only reachable when the release has none.
            files = item.get("wheels") or ([item["sdist"]] if item.get("sdist") else [])
            if not files:
                raise ValueError(f"{name}: lockfile names no artifact to audit")
            for file in files:
                kind = "wheel" if file["url"].endswith(".whl") else "sdist"
                url = file["url"]
                found.add(Artifact("pypi", name, item["version"], kind, url, file.get("size", 0)))
    else:
        data = json.loads(content)
        if data.get("lockfileVersion") not in (2, 3):
            raise ValueError(f"unsupported npm lockfile version: {path}")
        for location, item in data["packages"].items():
            # A bundled entry has no download of its own: its files ship inside
            # the parent tarball, which this audit reads in full.
            if not location or item.get("link") or item.get("inBundle"):
                continue
            name = item.get("name") or location.rsplit("node_modules/", 1)[-1]
            resolved = item.get("resolved", "")
            if not resolved.startswith(NPM_REGISTRY):
                raise ValueError(f"{name}: source {resolved!r} requires manual review")
            found.add(Artifact("npm", name, item.get("version", "unknown"), "npm", resolved))
    return found


def inventory(root: Path) -> set:
    """Collect artifacts from every lockfile tracked in the repository."""
    found = set()
    for path in sorted(root.rglob("uv.lock")) + sorted(root.rglob("package-lock.json")):
        if {"node_modules", ".venv"} & set(path.parts):
            continue
        found |= artifacts_from_lock(str(path), path.read_text())
    if not found:
        raise ValueError(f"no lockfiles found under {root}")
    return found


def metadata_license(text: str) -> str:
    """Read the license out of RFC 822 package metadata (METADATA, PKG-INFO)."""
    expression, legacy, classifiers = "", "", []
    for line in text.splitlines():
        if not line.strip():
            break  # headers end at the first blank line; the body is the README
        if line.startswith("License-Expression:"):
            expression = line.split(":", 1)[1].strip()
        elif line.startswith("License:"):
            legacy = line.split(":", 1)[1].strip()
        elif line.startswith("Classifier: License ::"):
            classifiers.append(line.split(":", 1)[1].strip())
    if expression or legacy:
        return expression or legacy
    aliases = {
        "License :: OSI Approved :: MIT License": "MIT",
        "License :: OSI Approved :: Apache Software License": "Apache-2.0",
        "License :: OSI Approved :: ISC License (ISCL)": "ISC",
        "License :: OSI Approved :: BSD License": "BSD-3-Clause",
        "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    }
    return aliases.get(classifiers[0], "") if len(classifiers) == 1 else ""


def npm_license(text: str) -> str:
    """Read the license from a package manifest, including the legacy list form."""
    declared = json.loads(text).get("license") or json.loads(text).get("licenses") or ""
    if isinstance(declared, list):
        parts = [part.get("type", "") if isinstance(part, dict) else part for part in declared]
        return " OR ".join(part for part in parts if part)
    if isinstance(declared, dict):
        return declared.get("type", "")
    return declared if isinstance(declared, str) else ""


def restricted_in(text: str) -> set:
    """Identify granted terms a permissive declaration would not cover."""
    upper = " ".join(text.upper().split())
    return {spdx for phrase, spdx in RESTRICTED_TEXTS if phrase in upper}


def scan_stream(stream) -> set | None:
    """Return the license notices a binary payload embeds, or None for text.

    Only an opaque binary can carry terms its package never declares. A shebang
    script or a data file is readable source whose license is the package's own,
    so it is not a vendored program at all and is reported as None. A real
    executable almost always embeds its own notice text: pandoc's carries the
    GPL. Chunks overlap so a phrase split across a boundary still matches.
    """
    head = stream.read(SCAN_CHUNK)
    if head.startswith(b"#!") or b"\x00" not in head[:4096]:
        return None
    found, tail = restricted_in(head.decode("latin-1")), head[-64:]
    while chunk := stream.read(SCAN_CHUNK):
        found |= restricted_in((tail + chunk).decode("latin-1"))
        tail = chunk[-64:]
        if len(found) == len(RESTRICTED_TEXTS):
            break
    return found


def is_program(name: str, mode: int) -> bool:
    """Pre-filter an executable payload by name, before its bytes decide.

    A license file carrying an executable bit is still a license file, and a
    Python extension module (.so/.pyd) is built from the package's own source.
    """
    if not mode & 0o111 or name.endswith("/") or PROGRAM_EXEMPT.search(name):
        return False
    if LICENSE_NAMES.search("/" + name):
        return False
    return Path(name).suffix.lower() in PROGRAM_SUFFIXES


def inspect_wheel(artifact: Artifact) -> Inspection:
    """Read a wheel's index and metadata in place, without downloading it."""
    size = artifact.size or remote_size(artifact.url)
    stream = io.BufferedReader(HttpRangeFile(artifact.url, size), buffer_size=1 << 18)
    result = Inspection()
    with zipfile.ZipFile(stream) as archive:
        for info in archive.infolist():
            name = info.filename
            if is_program(name, (info.external_attr >> 16) & 0o777):
                with archive.open(info) as member:
                    if (found := scan_stream(member)) is not None:
                        result.programs[name] = sorted(found)
            elif name.endswith(".dist-info/METADATA"):
                result.declared = metadata_license(
                    archive.read(info)[:MAX_TEXT_BYTES].decode("utf-8", "replace")
                )
            elif (
                LICENSE_NAMES.search("/" + name)
                and 0 < info.file_size <= MAX_TEXT_BYTES
                and (found := restricted_in(archive.read(info).decode("utf-8", "replace")))
            ):
                result.bundled[name] = sorted(found)
    return result


def inspect_tarball(artifact: Artifact) -> Inspection:
    """Read a source distribution or npm tarball, which cannot be read in place."""
    result = Inspection()
    with download(artifact.url) as payload, tarfile.open(fileobj=payload, mode="r:*") as archive:
        manifest = "package.json" if artifact.kind == "npm" else "PKG-INFO"
        for member in archive:
            if not member.isfile():
                continue
            # Both kinds nest everything under one directory: "package/" for
            # npm, "name-version/" for an sdist. Strip it so the manifest is
            # found at the path its specification gives.
            name = member.name.split("/", 1)[-1]
            if is_program(name, member.mode):
                if (found := scan_stream(archive.extractfile(member))) is not None:
                    result.programs[name] = sorted(found)
            elif name == manifest and not result.declared:
                text = (
                    archive.extractfile(member).read()[:MAX_TEXT_BYTES].decode("utf-8", "replace")
                )
                result.declared = (
                    npm_license(text) if artifact.kind == "npm" else metadata_license(text)
                )
            elif LICENSE_NAMES.search("/" + name) and 0 < member.size <= MAX_TEXT_BYTES:
                body = archive.extractfile(member).read().decode("utf-8", "replace")
                if found := restricted_in(body):
                    result.bundled[name] = sorted(found)
    return result


def inspect(artifact: Artifact, hosts: set) -> Inspection:
    """Read one artifact, refusing any host the policy does not name."""
    parts = urlparse(artifact.url)
    if parts.scheme != "https" or parts.netloc not in hosts:
        raise ValueError(f"artifact host requires review: {parts.netloc or artifact.url}")
    reader = inspect_wheel if artifact.kind == "wheel" else inspect_tarball
    return reader(artifact)


def normalize(expression: str) -> str:
    """Translate a known free-text license value into its SPDX equivalent."""
    if not isinstance(expression, str):
        return ""
    key = " ".join(expression.split()).rstrip(".").upper()
    if key in UNDECLARED:
        return ""
    return LEGACY_LICENSES.get(key, expression.strip())


def allowed_expression(expression, allowed) -> bool:
    """Decide whether an SPDX expression is satisfied by the allowed licenses."""
    expression = normalize(expression)
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 500:
        return False
    try:
        parsed = LICENSING.parse(expression, validate=True, strict=True)
        values = {
            symbol: LICENSING.TRUE if str(symbol) in allowed else LICENSING.FALSE
            for symbol in parsed.get_symbols()
        }
        return parsed.subs(values).simplify() == LICENSING.TRUE
    except (ExpressionError, ValueError):
        return False


def listing(names: list, limit: int = 6) -> str:
    """Name enough files to start a review without printing a whole archive."""
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, and {len(names) - limit} more"


def problems(found: Inspection, allowed: set) -> list:
    """Describe every reason an artifact needs a maintainer's decision."""
    issues = []
    if not allowed_expression(found.declared, allowed):
        issues.append(f"declared license needs review: {str(found.declared or 'UNKNOWN')[:160]!r}")
    for name, licenses in sorted(found.bundled.items()):
        if any(spdx not in allowed for spdx in licenses):
            issues.append(f"bundled license text in {name}: {', '.join(licenses)}")
    # One finding per package, not per file. A package that ships its own test
    # binaries would otherwise bury the packages that vendor someone else's.
    embedded, silent = {}, []
    for name, licenses in sorted(found.programs.items()):
        restricted = [spdx for spdx in licenses if spdx not in allowed]
        if restricted:
            embedded.setdefault(", ".join(restricted), []).append(name)
        else:
            silent.append(name)
    for licenses, names in sorted(embedded.items()):
        issues.append(f"bundles {len(names)} program(s) embedding {licenses}: {listing(names)}")
    if silent:
        issues.append(f"bundles {len(silent)} program(s) with no license text: {listing(silent)}")
    return issues


def approval(key: str, policy: dict):
    """Find a maintainer exception recorded for this exact package version."""
    return next(
        (
            entry
            for entry in policy["reviewed_packages"]
            if entry["package"] == key
            and entry.get("reason", "").strip()
            and entry.get("review_url", "").startswith("https://")
        ),
        None,
    )


def audit(artifacts, policy: dict, jobs: int = 16, reader=None, progress: int = 0) -> dict:
    """Inspect every artifact and group the findings by package version."""
    allowed = set(policy["allowed_licenses"])
    hosts = set(policy["artifact_hosts"])
    reader = reader or (lambda artifact: inspect(artifact, hosts))

    def review(artifact):
        try:
            return artifact, problems(reader(artifact), allowed)
        except Exception as error:  # noqa: BLE001 - unreadable artifacts fail closed
            return artifact, [f"could not be read: {type(error).__name__}: {error}"]

    findings, ordered, done = {}, sorted(artifacts), 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for artifact, issues in pool.map(review, ordered):
            done += 1
            if progress and done % progress == 0:
                print(f"  read {done}/{len(ordered)} artifacts", flush=True)
            for issue in issues:
                findings.setdefault(artifact.key, {}).setdefault(issue, []).append(
                    artifact.filename
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=".github/dependency-license-policy.json")
    parser.add_argument("--root", default=".", help="Repository root to audit")
    parser.add_argument("--jobs", type=int, default=16, help="Concurrent artifact reads")
    parser.add_argument("--report", help="Write all findings to this JSON file")
    args = parser.parse_args()

    policy = json.loads(Path(args.policy).read_text())
    artifacts = inventory(Path(args.root))
    print(f"Auditing {len(artifacts)} locked artifacts with {args.jobs} readers.", flush=True)

    started = time.time()
    hosts = set(policy["artifact_hosts"])
    findings = audit(
        artifacts,
        policy,
        args.jobs,
        reader=lambda artifact: inspect(artifact, hosts),
        progress=500,
    )
    print(f"Read every artifact in {time.time() - started:.0f}s.", flush=True)

    if args.report:
        Path(args.report).write_text(json.dumps(findings, indent=2, sort_keys=True))

    unresolved = {}
    for key, issues in sorted(findings.items()):
        if approved := approval(key, policy):
            print(f"REVIEWED {key}: {approved['review_url']}")
        else:
            unresolved[key] = issues

    for key, issues in sorted(unresolved.items()):
        for issue, files in sorted(issues.items()):
            print(f"REVIEW REQUIRED {key}: {issue} [{len(files)} file(s)]", file=sys.stderr)

    print(
        f"{len(artifacts)} artifacts audited, {len(findings)} package versions flagged, "
        f"{len(unresolved)} awaiting review."
    )
    return bool(unresolved)


if __name__ == "__main__":
    sys.exit(main())

# Bundled Ladybug JSON extension binaries

Ladybug's **Linux and Windows wheels dynamic-load the JSON extension**: at
runtime `INSTALL JSON` downloads `libjson.lbug_extension` from
`extension.ladybugdb.com`. Any offline, air-gapped, or firewall-restricted
deployment — or an outage of that server — leaves JSON-dependent features
(recall, temporal search) broken. macOS wheels currently compile it in
statically (verified on 0.16.0, 0.17.1, 0.18.2), but the 0.18 line *dropped*
static linking on Linux, so static linking is treated as fragile and macOS
binaries are bundled too.

Binaries placed here are loaded by absolute path (`LOAD EXTENSION '<path>'`),
which reads the file directly and never contacts the remote repo. The loading
ladder — by-name load first (static builds), then this bundle, then the remote
install as a last resort — lives in
`cognee_db_workers/_kuzu_helpers.py::load_json_extension`.

## Layout

Mirrors the extension repo, minus the trailing `json/` directory:

```
ladybug_extensions/
  v0.16.0/ ... v0.18.1/
    linux_amd64/libjson.lbug_extension    (~830 KB)
    linux_arm64/libjson.lbug_extension    (~880 KB)
    osx_amd64/libjson.lbug_extension      (~620 KB, insurance — macOS is static today)
    osx_arm64/libjson.lbug_extension      (~620 KB, insurance — macOS is static today)
    win_amd64/libjson.lbug_extension      (~13.4 MB — Windows links the lbug core in)
```

## How the right binary is chosen — no maintained mapping

There is deliberately **no version table to maintain anywhere**:

- **At runtime**, the engine announces its own requirement: the loader runs
  `INSTALL JSON FROM '<invalid local path>'`, which fails instantly (the path
  is treated as an unreachable URL — no network, verified ~0.01s offline on
  0.16.0–0.18.2) with an error naming the exact `<version>/<platform>` the
  installed binary requests (which can trail the package version: ladybug
  0.18.2 requests `v0.18.1`). Only that announced file is ever loaded.
  **Never hand-place or guess binaries: loading a wrong-version extension can
  segfault the process** — the probe is what makes selection safe.
- **At fetch time**, the ladybug constraint in `pyproject.toml` is the source
  of truth: the fetch script lists the extension repo's published version
  dirs and filters them through `scripts/ladybug_extension_versions.py`
  (range membership, plus the newest below-floor dir when the floor version's
  own dir trails below the range). Bumping the constraint automatically
  changes what ships; no other file needs editing.

The release workflows assert after `uv build` that every fetched version's
binary made it into the wheel, so a hollow wheel cannot ship silently. Guard
tests in `test_bundled_json_extension.py` pin the probe parser and the filter
semantics (cross-checked against `packaging`'s PEP 440).

## Populating

Binaries are not committed to git. Fetch the official ones with:

```bash
scripts/fetch_ladybug_json_extension.sh                    # everything pyproject supports
scripts/fetch_ladybug_json_extension.sh v0.18.1            # one version, all platforms
scripts/fetch_ladybug_json_extension.sh v0.18.1 linux_amd64 linux_arm64
```

The script pulls `ghcr.io/ladybugdb/extension-repo:latest` — the nginx image
serving as the origin behind `extension.ladybugdb.com` — and copies the
binaries out, so it works even while that server is unreachable. These are the
same official artifacts `INSTALL JSON` would download.

Wheels only contain the binaries present at `hatch build` time (the
`artifacts` entry in `pyproject.toml` lets the gitignored files in), so the
release pipeline runs the fetch script before building. The Docker image
copies every published Linux binary straight from the GHCR image in a build
stage — fully version-agnostic.

## Caveats

- The Linux binaries are glibc builds; musllinux (Alpine) ladybug wheels
  announce the same platform token but cannot dlopen them. The loader falls
  back to the remote install for that case.
- No `win_arm64` binary exists in the extension repo, so Windows ARM users
  stay on the by-name/remote path.

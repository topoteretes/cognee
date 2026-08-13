# Bundled Ladybug JSON extension binaries

Ladybug's macOS wheels compile the JSON extension into the native library, but
the **Linux and Windows wheels dynamic-load it**: at runtime `INSTALL JSON`
downloads `libjson.lbug_extension` from `extension.ladybugdb.com`. Any
offline, air-gapped, or firewall-restricted deployment — or any outage of that
server — leaves JSON-dependent features (recall, temporal search) broken.

Binaries placed here are loaded by absolute path (`LOAD EXTENSION '<path>'`),
which reads the file directly and never contacts the remote repo. The loading
ladder lives in `cognee_db_workers/_kuzu_helpers.py::load_json_extension`.

## Layout

Mirrors the extension repo, minus the trailing `json/` directory:

```
ladybug_extensions/
  v0.18.1/
    linux_amd64/libjson.lbug_extension
    linux_arm64/libjson.lbug_extension
    win_amd64/libjson.lbug_extension
```

The version directory is the **extension-repo version** the installed ladybug
requests, which can trail the package version (ladybug 0.18.2 requests
`v0.18.1`). The mapping from package version to repo version is
`_EXTENSION_REPO_VERSIONS` in `_kuzu_helpers.py` — extend it when bumping
ladybug: run `INSTALL JSON;` offline and read the version out of the error
URL.

## Populating

Binaries are not committed to git. Fetch the official ones with:

```bash
scripts/fetch_ladybug_json_extension.sh v0.18.1 linux_amd64 linux_arm64
```

The script pulls `ghcr.io/ladybugdb/extension-repo:latest` — the nginx image
serving as the origin behind `extension.ladybugdb.com` — and copies the
binaries out, so it works even while that server is unreachable. A
`win_amd64` binary also exists but is ~13.4 MB (Windows links the extension
against the full lbug core), so bundle it deliberately, not by default.

Release wheels only contain the binaries if this directory is populated before
`hatch build` runs — the `artifacts` entry in `pyproject.toml` lets the
gitignored files into the wheel. Docker images should populate it in the build
stage for the image's platform only.

## Caveats

- Binaries are glibc builds; musllinux (Alpine) ladybug wheels report the same
  platform token but cannot dlopen them. The loader falls back to the remote
  install for that case.
- macOS needs no binary here (statically linked), so nothing is bundled for
  `osx_*`.

# Bundled Ladybug JSON extension binaries

Ladybug's **Linux and Windows wheels dynamic-load the JSON extension**: at
runtime `INSTALL JSON` downloads `libjson.lbug_extension` from
`extension.ladybugdb.com`. Any offline, air-gapped, or firewall-restricted
deployment — or an outage of that server — leaves JSON-dependent features
(recall, temporal search) broken. macOS wheels compile it in statically
(verified on 0.16.0, 0.17.1, 0.18.2), but the 0.18 line *dropped* static
linking on Linux, so treat static linking as fragile.

Binaries placed here are loaded by absolute path (`LOAD EXTENSION '<path>'`),
which reads the file directly and never contacts the remote repo. The loading
ladder — by-name load first (static builds), then this bundle, then the remote
install as a last resort — lives in
`cognee_db_workers/_kuzu_helpers.py::load_json_extension`.

## Layout and version mapping

The layout mirrors the extension repo, minus the trailing `json/` directory:

```
ladybug_extensions/
  v0.18.1/
    linux_amd64/libjson.lbug_extension    (~830 KB)
    linux_arm64/libjson.lbug_extension    (~880 KB)
    osx_amd64/libjson.lbug_extension      (~620 KB, insurance — macOS is static)
    osx_arm64/libjson.lbug_extension      (~620 KB, insurance — macOS is static)
    win_amd64/libjson.lbug_extension      (~13.4 MB — Windows links the lbug core in)
```

The version directory is the **extension-repo version** the installed ladybug
requests, which can trail the package version. The verified mapping (each
entry discovered by running `INSTALL JSON;` offline and reading the URL out of
the error message) lives in `_EXTENSION_REPO_VERSIONS` in `_kuzu_helpers.py`:

| ladybug package | extension repo dir |
|---|---|
| 0.16.0 | v0.16.0 |
| 0.17.0, 0.17.1 | v0.17.0 |
| 0.18.0 | v0.18.0 |
| 0.18.1, 0.18.2 | v0.18.1 |

Only `v0.18.1` is shipped by default — it serves both ladybug versions cognee
actually resolves (0.18.1 via uv.lock, 0.18.2 via the pip range). When bumping
the ladybug constraint: extend the mapping, fetch the new version's binaries,
and ship those instead.

## Populating

Binaries are not committed to git. Fetch the official ones with:

```bash
scripts/fetch_ladybug_json_extension.sh v0.18.1            # all five platforms
scripts/fetch_ladybug_json_extension.sh v0.18.1 linux_amd64 linux_arm64
```

The script pulls `ghcr.io/ladybugdb/extension-repo:latest` — the nginx image
serving as the origin behind `extension.ladybugdb.com` — and copies the
binaries out, so it works even while that server is unreachable. These are the
same official artifacts `INSTALL JSON` would download.

Wheels only contain the binaries present at `hatch build` time (the
`artifacts` entry in `pyproject.toml` lets the gitignored files in), so the
release pipeline must run the fetch script before building. The Docker image
copies the Linux binaries straight from the GHCR image in a build stage.

## Caveats

- The Linux binaries are glibc builds; musllinux (Alpine) ladybug wheels
  report the same platform token but cannot dlopen them. The loader falls back
  to the remote install for that case.
- No `win_arm64` binary exists in the extension repo at v0.18.1, so Windows
  ARM users stay on the by-name/remote path.

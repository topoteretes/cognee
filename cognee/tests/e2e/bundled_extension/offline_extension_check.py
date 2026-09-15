"""In-container check: the bundled JSON extension must load with no network.

Runs inside a Linux container started with ``--network=none`` (see
run_offline_e2e.sh). Exercises the real ladder against a real engine:

1. A fresh connection's by-name ``LOAD`` tells us whether this wheel links
   JSON statically (macOS style). If so there is nothing to prove about the
   bundle on this platform — report and pass.
2. Otherwise the probe MUST resolve a bundled binary. This is the canary for
   the one silent-degradation risk of the bundling design: if a future
   ladybug changes the wording of its failing-INSTALL error, the probe stops
   matching and the loader silently falls back to the remote repo — invisible
   to the mocked unit tests, loud here.
3. The full ladder runs and a JSON function must work — offline, so a remote
   INSTALL cannot have papered over anything.
"""

import sys

sys.path.insert(0, "/app")

import ladybug

from cognee_db_workers._kuzu_helpers import bundled_json_extension_path, load_json_extension

print(f"ladybug version: {ladybug.__version__}", flush=True)

db = ladybug.Database("/tmp/offline_e2e_db")
conn = ladybug.Connection(db)

try:
    conn.execute("LOAD EXTENSION JSON;")
    static_build = True
except RuntimeError as error:
    if "not been installed" not in str(error):
        raise
    static_build = False

if static_build:
    print("JSON extension is statically linked in this wheel; bundle not needed here.", flush=True)
else:
    bundled = bundled_json_extension_path(conn.execute)
    print(f"bundled binary resolved: {bundled}", flush=True)
    assert bundled is not None, (
        "probe resolved no bundled binary — either the fetch step bundled the "
        "wrong version dirs, or this ladybug changed its failing-INSTALL error "
        "format and the probe regex in cognee_db_workers/_kuzu_helpers.py no "
        "longer matches (the silent-fallback-to-remote regression)."
    )
    load_json_extension(conn.execute)
    print("ladder completed via bundled binary", flush=True)

result = conn.execute("RETURN to_json([1,2,3]);")
rows = []
while result.has_next():
    rows.append(result.get_next())
print(f"to_json result: {rows}", flush=True)
assert rows == [["[1,2,3]"]], f"unexpected to_json output: {rows}"

print("OFFLINE-BUNDLED-EXTENSION-OK", flush=True)

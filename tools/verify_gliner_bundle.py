"""Verify an experimental wheel's RECORD and byte-for-byte upstream contents."""

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

from gliner_bundle import PREFIX, digest, file_digest


def verify(wheel_path, runtime_paths):
    """Verify against explicitly supplied local inputs, including their hashes."""
    inputs = {path.name: path for path in runtime_paths}
    with ZipFile(wheel_path) as wheel:
        manifest = json.loads(wheel.read(PREFIX + "BUNDLED_MANIFEST.json"))
        checked = 0
        for item in manifest:
            source = inputs[item["wheel"]]
            if file_digest(source) != item["sha256"]:
                raise ValueError(f"Input hash mismatch: {source.name}")
            with ZipFile(source) as original:
                for info in original.infolist():
                    if not info.is_dir():
                        if wheel.read(PREFIX + info.filename) != original.read(info):
                            raise ValueError(f"Bundled bytes changed: {info.filename}")
                        checked += 1
        records = [
            name
            for name in wheel.namelist()
            if name.count("/") == 1 and name.endswith(".dist-info/RECORD")
        ]
        if len(records) != 1:
            raise ValueError("Expected one top-level distribution RECORD")
        record = records[0]
        rows = list(csv.reader(io.StringIO(wheel.read(record).decode())))
        if {row[0] for row in rows} != set(wheel.namelist()) or len(rows) != len(wheel.namelist()):
            raise ValueError("RECORD does not enumerate all wheel members exactly once")
        for path, hash_value, size in rows:
            if path == record:
                continue
            data = wheel.read(path)
            if hash_value != "sha256=" + digest(data) or int(size) != len(data):
                raise ValueError(f"RECORD mismatch: {path}")
        return {
            "wheel": wheel_path.name,
            "bytes": wheel_path.stat().st_size,
            "sha256": file_digest(wheel_path),
            "vendor_files_verified": checked,
            "record_verified": True,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--runtime", type=Path, nargs=5, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.wheel, args.runtime), indent=2))

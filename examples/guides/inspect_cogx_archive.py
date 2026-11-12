"""Inspect portable memory before importing it, without a database or model.

Run with no arguments for a valid-archive and tampering demonstration, or pass
an existing unpacked COGX archive directory to inspect your own export.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from cognee.migration import COGXEntity, COGXFact, inspect_cogx_archive
from cognee.modules.migration.cogx import COGXArchiveWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", help="Existing unpacked COGX archive")
    args = parser.parse_args()
    if args.directory:
        report = inspect_cogx_archive(args.directory)
        print(report.model_dump_json(indent=2))
        raise SystemExit(0 if report.valid else 1)

    with TemporaryDirectory() as temporary:
        directory = Path(temporary) / "company_memory"
        with COGXArchiveWriter(directory) as writer:
            writer.write(COGXEntity(external_id="maya", name="Maya"))
            writer.write(COGXEntity(external_id="payments", name="Payments service"))
            writer.write(
                COGXFact(
                    external_id="ownership",
                    subject_ref="maya",
                    predicate="owns",
                    object_ref="payments",
                    fact_text="Maya owns the payments service.",
                )
            )

        report = inspect_cogx_archive(directory)
        print("Original archive:")
        print(report.model_dump_json(indent=2))
        assert report.valid
        assert report.record_counts == {"entity": 2, "fact": 1}

        # Modify only this demo's temporary fixture to simulate transfer damage.
        path = directory / "facts.jsonl"
        fact = json.loads(path.read_text(encoding="utf-8"))
        fact["fact_text"] = "Someone else owns the payments service."
        path.write_text(json.dumps(fact) + "\n", encoding="utf-8")

        report = inspect_cogx_archive(directory)
        print("\nAfter the demo deliberately alters a record:")
        print(report.model_dump_json(indent=2))
        assert not report.valid
        assert report.checksum_counts["invalid"] == 1


if __name__ == "__main__":
    main()

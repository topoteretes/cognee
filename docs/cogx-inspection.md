# Inspect a portable memory archive before importing

COGX exports can move a knowledge graph between Cognee instances. Inspect an
unpacked archive first to see what it contains and catch damaged or inconsistent
records before an import writes to your database.

```sh
cognee-cli inspect-archive backup_cogx
cognee-cli inspect-archive backup_cogx --json
```

The command runs locally, even if the global `--api-url` option is present.
It does not modify the archive, initialize databases, invoke an LLM or embedding
model, or import data. Normal package initialization and logging still apply.
Unpack `.cogx.tar.gz` files before using this command.

## Python API

```python
from cognee.migration import inspect_cogx_archive

report = inspect_cogx_archive("backup_cogx")
print(report.record_counts)
print(report.checksum_counts)
print(report.model_dump_json(indent=2))
if not report.valid:
    for diagnostic in report.diagnostics:
        print(diagnostic.file, diagnostic.line, diagnostic.code, diagnostic.message)
```

The report includes the declared COGX version and source system, actual and
declared record counts, checksum coverage, the presence of `permissions.json`,
and diagnostics identifying a file and, for record errors, its line number.
Diagnostics describe structural problems without printing record content.

## What is checked

- Manifest schema and compatibility with the reader's existing version policy.
- Each nonblank JSONL line against the existing typed COGX record models.
- Whether a typed record appears in the file for its record kind.
- Declared counts against the schema-valid records actually read.
- Existing record checksums using the same integrity calculation as the writer.
- Record-file paths remain within the archive directory.

Inspection continues after a malformed record so the report can identify other
problems in the same archive. It streams record files and retains counters and
diagnostics rather than the archive's record contents. A record with a valid
schema and a failed checksum is still included in its kind's record count.

The CLI exits with status **0** when there are no error diagnostics and **1**
otherwise. `--json` writes one JSON report to standard output in either case,
making it usable in scripts. Warnings do not change the exit status.

## Compatibility and limits of the report

Older archives may omit the manifest or checksums. A missing manifest produces
a warning, matching the existing reader's acceptance of manifest-less archives.
Missing checksums are reported as unverified coverage; they are not integrity
failures. Raw graph nodes also have no writer-stamped checksum. A directory with
neither a manifest nor a recognized record file is not an archive.

When a manifest explicitly supplies `counts`, that inventory is checked for
completeness: an observed record kind omitted from the inventory is a count
mismatch too. Legacy manifests that omit the `counts` field are accepted without
a count comparison.

`valid` means the inspected structure and available integrity checks passed.
It does not authenticate an archive's author, promise that unchecked records
have not changed, resolve fact references, predict mode-specific import results,
validate permissions, or guarantee that an import will succeed against a
particular database. Import behavior and defaults remain unchanged.

An export made with `include_permissions=True` can contain account information
and password hashes. The inspector only reports whether `permissions.json` is
present; it does not open or expose its contents. Inspecting an archive is not
permission to restore its accounts or grants.

## Runnable example

```sh
python examples/guides/inspect_cogx_archive.py
python examples/guides/inspect_cogx_archive.py /path/to/backup_cogx
```

The no-argument example creates a temporary archive, inspects it, deliberately
changes one fixture record, and shows the checksum failure. It needs no API key,
database, or model download.

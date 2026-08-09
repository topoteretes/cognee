# Task 5 Report: PostgreSQL and Neptune hybrid write paths

## Delivered

- PostgreSQL now creates `EdgeType_relationship_name` and `EdgeInstance_text`
  points from the shared edge-point builder in a single graph/vector data
  transaction. Relationship counts are queried after the graph upsert from the
  same session; both vector tables replace payload and vector on conflict.
- Neptune now writes graph edges first, queries current relationship counts, and
  writes one point set for each nonempty type and instance collection.
- Regression coverage checks distinct relationship-type/prose embeddings,
  graph-wide counts, shared instance IDs, and a rewritten instance vector.

## Validation

- `ruff check` passed for all changed source and test files.
- `ruff format --check` passed for all changed source and test files.
- `git diff --check` passed.
- Focused pytest could not start: `uv run` failed before test collection because
  its cache directory could not be created (`os error 183`).
- `python -m py_compile` could not start because the configured uv Python path
  is missing (`C:\\Users\\ACER\\AppData\\Roaming\\uv\\python\\cpython-3.11.14-windows-x86_64-none\\python.exe`).

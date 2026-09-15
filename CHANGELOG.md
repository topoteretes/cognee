# Changelog

## Unreleased

### Changed

- `GET /api/v1/datasets/{id}/data` now returns up to 100 documents by default.
  Clients must follow `limit` (1–1000) and `offset` (0–1,000,000) to read additional
  pages. The array response has no `X-Total-Count` header or truncation marker;
  use `GET /api/v1/datasets/{id}/data/count` to obtain the total. SDK `list_data`
  and explicit traversal clients follow pages and deduplicate overlaps. Offset
  traversal is not a snapshot: concurrent inserts/deletes may omit rows.
  The count is uncapped, but pages beyond offset 1,000,000 are rejected; datasets
  above 1,001,000 documents cannot be fully traversed through this endpoint.
  Full-traversal clients surface that error instead of returning partial results.

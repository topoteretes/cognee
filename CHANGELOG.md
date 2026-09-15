# Changelog

## Unreleased

### Changed

- `GET /api/v1/datasets/{id}/data` now returns up to 100 documents by default.
  Clients must follow `limit` (1–1000) and `offset` (0–1,000,000) to read additional
  pages. The array response has no `X-Total-Count` header or truncation marker;
  use `GET /api/v1/datasets/{id}/data/count` to obtain the total. SDK `list_data`
  and explicit full-traversal clients continue to return the full collection.

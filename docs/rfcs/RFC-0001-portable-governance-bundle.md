# RFC-0001: Portable Governance Bundle

## SECTION 1 — THE GAP

`cognee/api/v1/export/export.py` delegates to `export_dataset` in `cognee/modules/migration/export.py`, which exports graph structures in multiple formats: `pydantic`, `cogx`, `json`, `graphml`, and `cypher`. The `cogx` archive contains entities, documents, and facts (`COGXEntity`, `COGXDocument`, `COGXFact`).

It does NOT contain access control data. Specifically, the following tables are missing from the export:
- `acls` (`cognee/modules/users/models/ACL.py`)
- `permissions` (`cognee/modules/users/models/Permission.py`)
- `principals` (and its polymorphic types `tenants`, `roles`, `users` from `Tenant.py`, `Role.py`)
- `user_tenants` (`cognee/modules/users/models/UserTenant.py`)
- `role_default_permissions` (`cognee/modules/users/models/RoleDefaultPermissions.py`)
- `tenant_default_permissions` (`cognee/modules/users/models/TenantDefaultPermissions.py`)
- `user_default_permissions` (`cognee/modules/users/models/UserDefaultPermissions.py`)

The observability layer (`cognee/modules/observability/tracing.py`) records system operation spans using OpenTelemetry. It captures `COGNEE_DB_QUERY`, `COGNEE_PIPELINE_NAME`, and other trace signals, and holds them in `CogneeSpanExporter` via `trace_context.py`. However, there is no portable record of denied access. `check_permission_on_dataset.py` does not explicitly log authorization denials to a dedicated governance ledger; it merely executes `await get_specific_user_permission_datasets()` which may throw a `PermissionDeniedError`, but there is no explicit structural logging of this denied-action signal.

## SECTION 2 — GOVERNANCE BUNDLE SCHEMA

The bundle will be a single file named `governance.jsonld` inside the `cogx` archive. It is a JSON-LD document with `@context` pointing to:
- `https://www.w3.org/ns/odrl/2/` for policy expressions
- `https://schema.org/` for actor/timestamp metadata

It contains three top-level sections:

### 2a. permission_model
We map each real SQLAlchemy model to ODRL vocabulary:
- **Tenant** (`cognee/modules/users/models/Tenant.py`) → `odrl:Party` (with `@type: "odrl:Party"`, `uid: tenant.id`)
- **Dataset** → `odrl:Asset` (with `@type: "odrl:Asset"`, `uid: dataset.id`)
- **Permission** (`cognee/modules/users/models/Permission.py`) → `odrl:Action` (maps the `name` column string to an ODRL action URI).
- **ACL row** (`cognee/modules/users/models/ACL.py`) → `odrl:Policy` with `odrl:permission` containing:
  - `odrl:assigner` (the tenant/system granting access)
  - `odrl:assignee` (the `principal_id`)
  - `odrl:target` (the `dataset_id`)
  - `odrl:action` (the `permission_id` mapped to the string name)
  - `schema:startDate` (`created_at` from the ACL row)

**Mapping Cognee Action Strings to ODRL URIs:**
- `read` → `https://www.w3.org/ns/odrl/2/read`
- `write` → `https://www.w3.org/ns/odrl/2/modify`
- `delete` → `https://www.w3.org/ns/odrl/2/delete`
- For custom/non-standard actions (e.g. `manage`), we will use `odrl:use` as the base and extend it as `cognee:manage`.

### 2b. decision_history
Derived from existing OTEL trace signals (via `cognee/modules/observability/trace_context.py` and `CogneeTrace` attributes) where available.

Schema per record:
```json
{
  "actor_id": "uuid",
  "action": "string matching Permission.name values",
  "target_dataset_id": "uuid",
  "timestamp": "ISO 8601",
  "outcome": "ALLOWED",
  "policy_id": "uuid of the ACL/Permission row that authorized it",
  "trace_id": "OTEL trace id if available, null otherwise"
}
```

### 2c. rejection_trail
This does not exist today — new capture required.

Schema per record:
```json
{
  "actor_id": "uuid",
  "action": "string",
  "target_dataset_id": "uuid",
  "timestamp": "ISO 8601",
  "outcome": "DENIED",
  "denial_reason": "human-readable: e.g. no ACL entry for action=write on dataset=X",
  "policy_evaluated": "description of what was checked and found absent",
  "previous_hash": "SHA-256 of previous record or null for first record",
  "row_hash": "SHA-256(actor_id + action + target_dataset_id + outcome + timestamp + denial_reason + previous_hash)"
}
```
The hash inputs are concatenated exactly in the order specified above as UTF-8 strings before hashing to ensure determinism.

## SECTION 3 — OWL/AUTHORIZATION RECOMMENDATION (Option C)

**Option A**: Keep OWL purely knowledge-side, serialize ACL state as-is.
*Trade-off*: Portable but not machine-reasoned via standardized vocabularies.

**Option B**: Policy-as-ontology — OWL/ODRL reasoning at enforcement time.
*Trade-off*: Powerful but adds a heavyweight reasoner dependency to every single runtime access check.

**Option C (Recommended)**: RBAC/ACL stays the primary enforcement point (unchanged). Decisions are projected into the bundle as ODRL records (portable, standards-aligned, and optionally reasoned offline). OWL stays strictly knowledge-side.

**Justification**: `RDFLibOntologyResolver` in `cognee/modules/ontology/rdf_xml/RDFLibOntologyResolver.py` purely handles knowledge-side entity grounding and parsing (e.g., `find_closest_match`, `get_subgraph`). It does not touch authorization at all. Option C does not change this. 
Option B would require us to inject RDFLib parsing and graph execution overhead directly into the hot-path of `check_permission_on_dataset.py` before any dataset read/write occurs, crippling performance. Option C is the optimal upgrade path.

## SECTION 4 — ROUND-TRIP GUARANTEE

After `import_governance_bundle()` (to be implemented via `cognee/modules/migration/import_source.py` detection) completes successfully on a fresh instance, `check_permission_on_dataset(user, permission_type, dataset_id)` returns the exact same result as on the exporting instance, for every `(actor_id, action, dataset_id)` triple present in the exported `permission_model` section.

**Caveats**:
- User UUIDs must be globally stable across instances.
- Dataset UUIDs must be globally stable across instances.
- The importing instance must have the exact same `Permission.name` vocabulary established.

## SECTION 5 — WHAT DOES NOT CHANGE

- The existing `export_dataset()` function in `cognee/modules/migration/export.py` will not have its graph extraction formats altered.
- The governance bundle is merely an ADDITIONAL sidecar file (`governance.jsonld`) inside the `cogx` archive.
- All existing `cogx` consumers (`COGXArchiveWriter`, `MemorySource`) are unaffected.
- The RBAC enforcement hot path in `check_permission_on_dataset.py` remains completely unchanged in structure.

## SECTION 6 — FILE CHANGES OVERVIEW

Files that will be touched in Chunks 2, 3, and 4:
- `cognee/modules/migration/export.py`: Modify `_write_cogx` or `export_dataset` to synthesize and write `governance.jsonld` into the archive directory.
- `cognee/modules/migration/import_source.py`: Add a parsing step to detect `governance.jsonld` and pipe it to the governance importer logic.
- `cognee/alembic/versions/[id]_add_audit_event_table.py`: Create the schema for the `rejection_trail`.
- `cognee/modules/users/permissions/methods/check_permission_on_dataset.py`: Attach an async hook to capture and write `DENIED` events to the audit table on `PermissionDeniedError`.
- `cognee/modules/governance/*`: Create the new models, serializers, and hash-chain logic for constructing and validating the bundle.

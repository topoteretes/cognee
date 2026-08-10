---
name: cognee-permissions
description: Use when working with cognee's permission system — understanding or changing how users, roles, and tenants get access to datasets, how ACL grants work, where permissions are enforced in add/cognify/search/delete, and how the grant records surface in the memory-provenance view.
---

# The cognee permission system

## The master switch

`ENABLE_BACKEND_ACCESS_CONTROL` decides whether any of this runs:

- `true` (default): multi-tenant mode. Every API call requires auth, every
  dataset operation is permission-checked, and each user+dataset pair gets
  isolated graph/vector/relational databases (tracked in the
  `DatasetDatabase` model, supported backends: Kuzu, LanceDB, SQLite,
  Postgres).
- `false`: single-user local mode. No auth, shared databases, every
  permission check short-circuits to allowed. Nothing below applies.

## The core model: principals, permissions, ACL grants

Everything reduces to one relation — **a grant**: *principal* × *permission*
× *dataset*, stored as one `ACL` row (`modules/users/models/ACL.py`).

- **Principal** (`Principal.py`) is polymorphic: `User`, `Role`, and
  `Tenant` all inherit from it. Any of the three can hold a grant, which is
  how role-wide and tenant-wide access work — one ACL row covers every
  member.
- **Permission** (`Permission.py`) is one of exactly four names, defined in
  `permissions/permission_types.py`: `read`, `write`, `delete`, `share`.
  `share` is the meta-permission: it gates granting/revoking access for
  others.
- **Membership** is separate from grants: `UserRole` and `UserTenant` link
  users into roles/tenants. A user's effective access is the union of their
  own grants and the grants of every role/tenant they belong to.

## How grants come into existence

1. **Dataset creation** (`modules/data/methods/create_authorized_dataset.py`):
   the creating user is granted **all four permissions** on the new dataset.
   If the user has a `parent_user_id` (sub-users/agent identities), the
   parent is auto-granted all four as well — parents always see their
   children's datasets.
2. **Explicit sharing** (`permissions/methods/
   authorized_give_permission_on_datasets.py`): the caller must hold
   `share` on the target datasets, then any principal (user, role, or
   tenant) can be granted any permission. Revocation mirrors this
   (`authorized_revoke_permission_on_datasets.py`).
3. **Default permissions** (currently dormant scaffolding): the
   `UserDefaultPermissions`, `RoleDefaultPermissions`, and
   `TenantDefaultPermissions` tables + `give_default_permission_to_*`
   methods exist, but as of now nothing in the runtime calls or consults
   them — no flow reads these tables when resolving access. Treat them as
   the intended home for standing permission sets, not active behavior.

## Where permissions are enforced

The single chokepoint for dataset resolution is
`get_authorized_existing_datasets(datasets, permission, user)` — every
entrypoint resolves names/IDs through it with the permission it needs:

| Operation | Required permission | Enforcement path |
|---|---|---|
| `add` / `cognify` / `remember` | `write` | dataset resolution before the pipeline runs |
| `search` / `recall` / visualize | `read` | dataset resolution + `get_document_ids_for_user` filters retrievable documents per-dataset |
| `delete` / prune of a dataset | `delete` | `datasets.py` resolves with `"delete"` |
| grant/revoke for others | `share` | `authorized_give/revoke_permission_on_datasets` |

Two behaviors worth knowing:

- **Denied reads return empty results, not 403.** A search against a
  dataset you cannot read yields `[]` — deliberate, to avoid leaking which
  datasets exist. When debugging "search returns nothing", check grants
  before checking the graph.
- Fine-grained document filtering (`get_document_ids_for_user`) runs inside
  the search path: it collects the datasets readable by the user (own +
  role + tenant grants) and restricts retrieval to documents of those
  datasets.

## Roles, tenants, and who may manage them

- **User management** (listing tenant users, assigning/removing roles,
  adding/removing users) is allowed for the **tenant owner** always, and
  for members of roles named in `USER_MANAGEMENT_ALLOWED_ROLE_NAMES`
  (currently `{"admin"}`) — extend that frozenset to introduce new admin
  role names (`permissions/permission_types.py`).
- **Role visibility**: members of a role can see the role itself and their
  co-members; anyone with user-management permission sees all
  (`tenants/methods/get_users_in_role.py`). Lookups are tenant-scoped — a
  role id from another tenant cannot be used to read that tenant's members.

## The grant records in memory provenance (the new grant view)

`api/v1/visualize/memory_provenance.py` surfaces the ACL grants as
first-class graph data. Each grant becomes an `AclGrantRecord`:

```python
{"principal_id": ..., "principal_kind": "user" | "role" | "tenant", "permission": ...}
```

and is rendered into the provenance graph as an edge from the principal
node to the dataset, with the permission mapped to a relation name
(`_ACL_EDGE_RELATIONS`):

| permission | provenance edge |
|---|---|
| read | `reads` |
| write | `writes` |
| delete | `can_delete` |
| share | `can_share` |

Grants are rendered (never dropped) even when the principal is unknown,
because "an ACL row exists because someone granted it". The view is exposed
through the schema router (`get_schema_router.py`):
`visualize_memory_provenance` (HTML) and `get_memory_provenance_payload`
(JSON) — this is where you *see* the permission state of a memory rather
than query it.

## HTTP API surface (`api/v1/permissions/routers/get_permissions_router.py`)

| Endpoint | What it does |
|---|---|
| `POST /permissions/datasets/{principal_id}` | grant a permission on datasets to a principal (requires `share`) |
| `DELETE /permissions/datasets/{principal_id}` | revoke a permission |
| `POST /permissions/roles` · `DELETE /permissions/roles/{role_id}` | create/delete a role |
| `POST/DELETE /permissions/users/{user_id}/roles` | add/remove a user to/from a role |
| `POST /permissions/users/{user_id}/tenants` | add a user to a tenant |
| `GET /permissions/tenants/{tenant_id}/roles/{role_id}/users` | members of a role (self-visible to members) |
| `GET /permissions/tenants/{tenant_id}/roles/users/{user_id}` | a user's roles |
| `GET /permissions/tenants/{tenant_id}/users` | users in a tenant |
| `GET /permissions/tenants/me` | the caller's tenants |

## Key files map

- Models: `cognee/modules/users/models/` — `ACL`, `Principal`, `Permission`,
  `Role`, `Tenant`, `UserRole`, `UserTenant`, `*DefaultPermissions`,
  `DatasetDatabase`
- Methods: `cognee/modules/users/permissions/methods/` — grant/revoke,
  checks, dataset resolution, document filtering
- Enforcement chokepoint: `cognee/modules/data/methods/`
  (`get_authorized_existing_datasets`, `create_authorized_dataset`)
- Grant provenance view: `cognee/api/v1/visualize/memory_provenance.py`
- HTTP API: `cognee/api/v1/permissions/routers/get_permissions_router.py`

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
- `false`: single-user mode. Permission checks short-circuit to allowed,
  there is no per-dataset isolation, and **every user's operations resolve
  to the same shared databases and datasets**. Authentication is a separate
  knob: `REQUIRE_AUTHENTICATION`. Unset, it inherits this switch (so
  turning access control off also turns auth off) — but if
  `REQUIRE_AUTHENTICATION=true` is set, endpoints still demand a login;
  authenticated users are identified but *not isolated*, all pointing at
  the same data. The reverse misconfiguration
  (`REQUIRE_AUTHENTICATION=false` with access control on) is ignored: auth
  is forced on with a warning, because multi-tenant isolation is
  meaningless without identity (`get_authenticated_user.py`).

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
3. **Capabilities — tenant-scoped grants of actions, not data** (landing
   via PR #4302, currently in review): a new `principal_capabilities`
   table, keyed on `(principal, tenant, capability)`. Where an ACL row
   grants access to *a dataset*, a capability grants *an action inside a
   tenant* — the first one being `manage_users`. The catalog of capability
   names is code (`CAPABILITY_TYPES` in `permission_types.py`), not a
   database table, "because the code is what gives each name meaning";
   only the assignment of a capability to a principal is data. `tenant_id`
   is stored on every row because a user can belong to multiple tenants:
   it pins each grant to the user's membership in one specific tenant, so
   holding a capability in one tenant never carries over to the same
   user's other tenants. Resolution
   (`get_effective_capabilities(user, tenant)`) returns the union of what
   the tenant grants all of its members, what the user's roles in that
   tenant grant, and what the user was granted personally — there is no
   deny in the model, resolution is gated on actual tenant membership, and
   the tenant owner short-circuits as holding every capability.
   Grant/revoke endpoints ride the permissions router and are gated by
   capabilities of their own: granting needs `grant_capabilities`,
   revoking needs `revoke_capabilities`, and neither comes with
   `manage_users`. A granter can only pass on capabilities they hold
   themselves (`get_unheld_capabilities`), the same rule role assignment
   follows, so `grant_capabilities` is not a way to reach the rest of the
   catalog. The owner and the `admin` role hold everything. Every capability check goes through
   `has_grant_permission(requester, tenant, capability)`;
   `has_user_management_permission` is that check for `manage_users`.
   Each row records who made the grant in `granted_by` (nullable, kept
   when the granter is removed), and both endpoints take `capability`
   repeated to grant or revoke several at once, all or nothing. For a user
   principal the grant lands in the `tenant_id` given, or the caller's
   current tenant; a role or tenant principal always uses its own tenant.
   A missing principal or tenant answers like a refusal (403), so the
   endpoints do not reveal which ids exist. Removing a user from a tenant
   drops their personal capabilities there, and deleting a role drops the
   role's, so neither comes back later.

## Where permissions are enforced

The single chokepoint for dataset resolution is
`get_authorized_existing_datasets(datasets, permission, user)` — every
entrypoint resolves names/IDs through it with the permission it needs:

| Operation | Required permission | Enforcement path |
|---|---|---|
| `add` / `cognify` / `remember` | `write` | dataset resolution before the pipeline runs |
| `search` / `recall` / visualize | `read` | dataset resolution; retrieval is restricted to documents of readable datasets |
| `delete` / prune of a dataset | `delete` | `datasets.py` resolves with `"delete"` |
| grant/revoke for others | `share` | `authorized_give/revoke_permission_on_datasets` |

Two behaviors worth knowing:

- **Denied reads return empty results, not 403.** A search against a
  dataset you cannot read yields `[]` — deliberate, to avoid leaking which
  datasets exist. When debugging "search returns nothing", check grants
  before checking the graph.

## Roles, tenants, and who may manage them

- **User management** (listing tenant users, assigning/removing roles,
  adding/removing users) is allowed for the **tenant owner** always, and
  today for members of roles named in `USER_MANAGEMENT_ALLOWED_ROLE_NAMES`
  (currently `{"admin"}`, `permissions/permission_types.py`). That
  name-matching is a known footgun — any customer group that happens to be
  called "admin" gets user management — and PR #4302 replaces it: the
  check becomes "does the requester hold the `manage_users` capability in
  this tenant" (owner always passes), with the role-name match kept only
  as a deprecated fallback so tenants upgrading from the old check don't
  lose user management until their `admin` role is granted the capability.
  The fallback sits in `has_grant_permission`, so an `admin` role also
  passes the grant and revoke checks until it is migrated.
- **Creating roles, assigning them and adding users to a tenant** need
  `manage_users`, not ownership. Assigning a role has one more rule
  (`require_role_capabilities`): the requester must hold every capability
  the role carries, and a role named `admin` counts as carrying all of
  them. Without it, `manage_users` would reach every other capability by
  joining a role that has it.
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
| `GET /permissions/tenants/{tenant_id}/roles/users/{user_id}` | a user's roles in that tenant (404 if not a member) |
| `GET /permissions/tenants/{tenant_id}/users` | users in a tenant |
| `GET /permissions/tenants/me` | the caller's tenants |

## Key files map

- Models: `cognee/modules/users/models/` — `ACL`, `Principal`, `Permission`,
  `Role`, `Tenant`, `UserRole`, `UserTenant`, `DatasetDatabase` (and
  `PrincipalCapability` once #4302 lands)
- Methods: `cognee/modules/users/permissions/methods/` — grant/revoke,
  checks, dataset resolution, document filtering
- Enforcement chokepoint: `cognee/modules/data/methods/`
  (`get_authorized_existing_datasets`, `create_authorized_dataset`)
- Grant provenance view: `cognee/api/v1/visualize/memory_provenance.py`
- HTTP API: `cognee/api/v1/permissions/routers/get_permissions_router.py`

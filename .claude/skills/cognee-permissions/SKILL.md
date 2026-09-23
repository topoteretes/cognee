---
name: cognee-permissions
description: Use when working with cognee's users, permissions, and multi-tenancy — creating users, tenants and roles, sharing datasets (read/write/delete/share grants), acting as a specific user in the SDK or authenticating over HTTP (login, API keys), turning access control on or off, debugging PermissionDeniedError or missing datasets, or understanding where permissions are enforced and how datasets are isolated.
---

# Users, permissions, and multi-tenancy

Every dataset belongs to an owner, and every operation on it is checked
against a **grant**: a principal (user, role, or tenant) holding a
permission (`read`, `write`, `delete`, `share`) on a dataset. With access
control on (the default), each user+dataset pair also gets its own graph and
vector databases.

## Use it

### Act as a user in the SDK

Without `user=`, SDK calls run as the default user
(`default_user@example.com`, or `DEFAULT_USER_EMAIL`). To act as someone
else, create or load them and pass `user=`:

```python
import cognee
from cognee.modules.users.methods import create_user, get_user

alice = await create_user("alice@example.com", "password")
bob = await create_user("bob@example.com", "password")

res = await cognee.remember("Alice's notes", dataset_name="alice_notes", user=alice)
await cognee.recall("What are the notes about?", user=alice, datasets=["alice_notes"])
```

`remember`, `recall`, `search`, `forget`, `improve` and the dataset helpers
all take `user=`.

### Share a dataset

The caller must hold `share` on the dataset. Share **by dataset id**:

```python
from cognee.modules.users.permissions.methods import (
    authorized_give_permission_on_datasets,
    authorized_revoke_permission_on_datasets,
)

await authorized_give_permission_on_datasets(
    bob.id,                 # principal: a user, role, or tenant id
    [res.dataset_id],       # dataset ids
    "read",                 # "read" | "write" | "delete" | "share"
    alice.id,               # the owner making the grant
)

await cognee.recall("...", user=bob, dataset_ids=[res.dataset_id])   # by id
```

Bob must address Alice's dataset by **id**: dataset *names* resolve only
among the caller's own datasets.

### Tenants and roles (organizations and groups)

```python
from cognee.modules.users.tenants.methods import add_user_to_tenant, create_tenant, select_tenant
from cognee.modules.users.roles.methods import add_user_to_role, create_role

tenant_id = await create_tenant("Acme", alice.id)            # alice owns it
await select_tenant(user_id=alice.id, tenant_id=tenant_id)
role_id = await create_role(role_name="Researcher", owner_id=alice.id)
await add_user_to_tenant(user_id=bob.id, tenant_id=tenant_id, owner_id=alice.id)
await add_user_to_role(user_id=bob.id, role_id=role_id, owner_id=alice.id)
await select_tenant(user_id=bob.id, tenant_id=tenant_id)

alice = await get_user(alice.id)   # reload after changing the active tenant
res = await cognee.remember(text, dataset_name="acme_docs", user=alice)
await authorized_give_permission_on_datasets(role_id, [res.dataset_id], "read", alice.id)
```

- A user acts inside one **active tenant** (`select_tenant`; `None` is the
  personal space). Datasets are created in the active tenant.
- You can only grant a role or tenant access to datasets **in that
  tenant**. A dataset created in the personal space cannot be shared with a
  tenant's role; recreate it with the tenant active.
- A role's members get the role's grants; a tenant's members get the
  tenant's grants.

Full walkthrough: `examples/demos/permissions/user_permissions_and_access_control_example.py`
(also `tenant_role_setup_example.py`, `tenant_role_constraints_example.py`).

### Over HTTP

- **Register / log in:** `POST /api/v1/auth/register`, then
  `POST /api/v1/auth/login` (form fields `username`, `password`). The
  response sets an auth cookie and returns `{"access_token", "token_type":
  "bearer"}`; send `Authorization: Bearer <token>`.
- **API keys:** `POST /api/v1/auth/api-keys` creates one (`GET` lists,
  `DELETE /api-keys/{id}` removes); send it as `X-Api-Key: <key>`.
- **Default user over HTTP:** it has no password unless
  `DEFAULT_USER_PASSWORD` is set (the server logs a warning at startup).
- **Permissions routes** (`/api/v1/permissions`):

| Endpoint | What it does |
|---|---|
| `POST /datasets/{principal_id}?permission_name=read` + JSON body `[dataset_ids]` | Grant (needs `share`) |
| `DELETE /datasets/{principal_id}` | Revoke |
| `GET /principals/{principal_id}/datasets?permission_name=read` | Datasets a principal holds a permission on |
| `POST /tenants` · `POST /tenants/select` · `GET /tenants/me` | Create, switch, list your tenants |
| `POST /users/{user_id}/tenants` · `DELETE /tenants/{tenant_id}/users/{user_id}` | Add/remove a tenant member |
| `GET /tenants/{tenant_id}/users` | Tenant members |
| `POST /roles` · `DELETE /roles/{role_id}` · `GET /tenants/{tenant_id}/roles` | Manage roles |
| `POST` / `DELETE /users/{user_id}/roles` | Add/remove a role member |
| `GET /tenants/{tenant_id}/roles/{role_id}/users` · `GET /tenants/{tenant_id}/roles/users/{user_id}` | Role members; a user's roles (404 if not a member) |

### Turning access control off

`ENABLE_BACKEND_ACCESS_CONTROL` is the master switch:

- `true` (default): multi-tenant. API calls require auth, every dataset
  operation is permission-checked, and each user+dataset gets isolated graph
  and vector databases.
- `false`: single-user. Permission checks always pass, there is no
  isolation, and **every user reads and writes the same shared databases**.
  Use it only for a single-user deployment.

Authentication follows the switch unless `REQUIRE_AUTHENTICATION` is set.
`REQUIRE_AUTHENTICATION=true` with access control off keeps logins but not
isolation; `REQUIRE_AUTHENTICATION=false` with access control on is ignored
(auth is forced on with a warning).

> **For production multi-tenant deployments** (managed isolation, the
> production Postgres adapter, and horizontal scaling), contact
> social@cognee.ai.

## Pitfalls

- **Denied is not the same as empty.**
  - Asking for a dataset **id** you cannot read raises
    `PermissionDeniedError` (HTTP 403).
  - Asking for a dataset **name** that is not yours raises
    `DatasetNotFoundError`, even if it was shared with you. Use the id.
  - Passing **no datasets** searches only what you can read, so a user with
    no grants simply gets `[]`.
- **Grants need `share`**, and role/tenant grants need the dataset to be in
  that tenant; otherwise `PermissionDeniedError`.
- **Reload the user after `select_tenant`** (`get_user(id)`): an old `User`
  object still carries the previous active tenant.
- **Same name, different datasets.** Dataset ids are per owner and tenant,
  so two users' `"notes"` datasets are unrelated.
- **Unsupported backends are a hard error.** With access control on, both
  the graph and vector backends need a dataset-database handler. If either
  lacks one (e.g. Neptune, Neptune Analytics, most community vector
  adapters), cognee raises `OSError` naming it
  (`multi_user_support_possible()` in `cognee/context_global_variables.py`),
  never a silent fall back to shared databases. Switch backends or set
  `ENABLE_BACKEND_ACCESS_CONTROL=false`. The support matrix is in CLAUDE.md
  ("Multi-Tenant Access Control").
- **User management by role name.** Tenant owners, and members of roles
  named `admin` (`USER_MANAGEMENT_ALLOWED_ROLE_NAMES`), can manage tenant
  users. Any group that happens to be called "admin" gets that power.

## How it works

### The model

A grant is one `ACL` row: principal × permission × dataset
(`cognee/modules/users/models/ACL.py`).

- **Principal** (`Principal.py`) is polymorphic: `User`, `Role`, and `Tenant`
  all inherit from it, so one ACL row can cover every member of a role or
  tenant.
- **Permission** is one of four names (`permissions/permission_types.py`):
  `read`, `write`, `delete`, `share`. `share` gates granting and revoking.
- **Membership** (`UserRole`, `UserTenant`) is separate from grants. A
  user's access is the union of their own grants and those of their roles
  and tenants.

Grants come from:

1. **Dataset creation** (`cognee/modules/data/methods/create_authorized_dataset.py`):
   the creator gets all four permissions. If the creator has a
   `parent_user_id` (a sub-user or agent identity,
   `create_user(..., parent_user_id=...)`), the parent gets all four too.
2. **Explicit sharing** (`authorized_give_permission_on_datasets` /
   `authorized_revoke_permission_on_datasets`).

### Where it is enforced

Every entry point resolves datasets through
`get_authorized_existing_datasets(datasets, permission, user)`
(`cognee/modules/data/methods/`):

| Operation | Permission |
|---|---|
| `remember` / `add` / `cognify` / `improve` | `write` |
| `recall` / `search` / visualize | `read` |
| `forget` / delete / empty a dataset | `delete` |
| grant / revoke | `share` |

### Isolation

With access control on, each user+dataset pair has its own graph and vector
databases, recorded in the `DatasetDatabase` model (names, providers,
handlers, connection info, migration revision). The relational database
(users, ACLs, the registry) is always shared. The handler is chosen from the
configured providers; the registry is
`cognee/infrastructure/databases/dataset_database_handler/supported_dataset_database_handlers.py`.
The `*_shared` handlers (`pgvector_shared`, `postgres_graph_shared`) give
each dataset its own Postgres *schema* inside cognee's main database instead
of a separate database, so no `CREATE DATABASE` privilege is needed. Select
them with `VECTOR_DATASET_DATABASE_HANDLER` / `GRAPH_DATASET_DATABASE_HANDLER`.

### Seeing grants

`cognee/api/v1/visualize/memory_provenance.py` renders ACL grants as edges
from principal to dataset (`reads`, `writes`, `can_delete`, `can_share`),
served by the schema router (`visualize_memory_provenance` HTML,
`get_memory_provenance_payload` JSON).

### Key files

- Models: `cognee/modules/users/models/` (`ACL`, `Principal`, `Permission`,
  `Role`, `Tenant`, `UserRole`, `UserTenant`, `DatasetDatabase`,
  `UserApiKey`, and the `*DefaultPermissions` models)
- Users, tenants, roles: `cognee/modules/users/methods/`,
  `cognee/modules/users/tenants/methods/`, `cognee/modules/users/roles/methods/`
- Grants and checks: `cognee/modules/users/permissions/methods/`
- Auth: `cognee/modules/users/authentication/`,
  `cognee/api/v1/users/routers/`, `cognee/api/v1/api_keys/routers/`
- HTTP permissions API: `cognee/api/v1/permissions/routers/get_permissions_router.py`

## Extending it

- **New operation on a dataset:** resolve it through
  `get_authorized_existing_datasets` with the right permission before doing
  any work; never read a dataset by id without that check.
- **New backend:** implement a `DatasetDatabaseHandlerInterface` and register
  it in the handler registry (or at runtime with
  `use_dataset_database_handler()`), otherwise multi-tenant mode refuses to
  start with it.
- **Not merged yet: capabilities (PR #4302).** A planned
  `principal_capabilities` table would grant tenant-scoped actions (first
  `manage_users`) instead of matching the `admin` role name. None of it is
  on `dev`; do not write code against it until it lands.
- Tests: `cognee/tests/unit/users/`, `cognee/tests/unit/modules/users/`, and
  the examples above.

# Tenants

> Organization-level access control and permission inheritance in Cognee

# Tenants

A tenant represents an organization or group. [Tenants](./tenants) contain [users](./users) and can be granted permissions on [datasets](./datasets), which apply to all members. This enables organization-wide access control and simplifies permission management for teams.

<Info>**Tenant-level permissions** — When a tenant is granted a permission on a dataset, all users in that tenant automatically inherit that permission.</Info>

## Tenant Concept

[Tenants](./tenants) are created by [users](./users) who become the tenant owner. The owner can add other [users](./users) to the tenant. [Users](./users) can belong to at most one tenant, but [tenants](./tenants) can contain multiple [users](./users).

## Tenant-Level Permissions

When a [tenant](./tenants) is granted a permission on a [dataset](./datasets), all [users](./users) in that tenant automatically inherit that permission. This happens through the permission checking mechanism: `get_all_user_permission_datasets()` unions the [user](./users)'s direct permissions with their [tenant](./tenants)'s permissions.

[Tenants](./tenants) start with zero permissions. You can leave the tenant principal empty and manage access purely through individual [user](./users) permissions, or grant tenant-wide permissions for organization-wide resources.

## Permission Inheritance

Tenant-level grants are blanket: once a [dataset](./datasets) permission is assigned to the tenant principal, every [user](./users) whose `tenant_id` matches inherits it. [Users](./users) can also receive direct permissions that differ from the tenant defaults, giving you flexibility to customize access for specific [users](./users) within the same tenant.

<Accordion title="Tenant Model Fields">
  The Tenant model defines what gets stored in the SQL database. The `tenants` table contains:

  * `id`: Unique identifier (UUID primary key, references principals.id)
  * `name`: Human-readable name (unique)
  * `owner_id`: ID of the [user](./users) who created the tenant
</Accordion>

<Accordion title="Tenant Creation">
  * `create_tenant(tenant_name, user_id)`: Creates a new tenant with the specified [user](./users) as owner
  * `add_user_to_tenant(user_id, tenant_id, owner_id)`: Adds an existing [user](./users) to a tenant (owner only)
</Accordion>

<Accordion title="Limitations">
  * [Users](./users) without a tenant exist but are isolated
  * API endpoints for tenant management
</Accordion>

## Tenant Management

Tenant owners can:

* Add [users](./users) to the tenant
* Remove [users](./users) from the tenant
* Grant permissions to the tenant principal
* Manage tenant-level access to [datasets](./datasets)

## Use Cases

Tenants are ideal for:

* **Organization-wide access** — Grant broad permissions to all team members
* **Department-level isolation** — Keep different departments' data separate
* **Project-based grouping** — Organize users around specific projects or initiatives
* **Scalable permission management** — Avoid granting individual permissions to many users

## Data Isolation

Each tenant's data is completely isolated:

* **Database separation** — Each user's data is stored in their own directory
* **Permission boundaries** — [Users](./users) can only access [datasets](./datasets) they have permissions for
* **No cross-tenant access** — Data from one tenant cannot be accessed by [users](./users) from another tenant

## Best Practices

* **Start with tenant-level permissions** — Grant broad access at the tenant level
* **Refine with [user](./users) permissions** — Override tenant defaults for specific [users](./users) when needed
* **Use [roles](./roles) for granular control** — Create [roles](./roles) within tenants for more specific access patterns
* **Regular permission audits** — Review and update permissions as team structure changes

<Columns cols={2}>
  <Card title="Roles" icon="users" href="/core-concepts/permissions-system/roles">
    Learn about role-based permissions within tenants
  </Card>

  <Card title="ACL" icon="shield" href="/core-concepts/permissions-system/acl">
    Understand how permissions are stored and checked
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
# Principals

> The unified abstraction for entities that can hold permissions in Cognee

# Principals: The Abstraction

A principal is any entity that can hold permissions in Cognee. This abstraction allows the permission system to work with different types of entities in a unified way, eliminating the need for separate permission systems for users, tenants, and roles.

<Info>**Polymorphic design** — All principal types use the same permission mechanism, making the system flexible and consistent.</Info>

## Principal Types

There are three types of principals:

* **[Users](./users)** — Individual people who interact with the system
* **[Tenants](./tenants)** — Organizations or groups that contain users
* **[Roles](./roles)** — Groups of users within a tenant

All three types inherit from the same base Principal class, which means they can all be granted permissions on datasets using the same functions and mechanisms.

## How Principals Work with Permissions

The system stores permissions by linking principals to datasets. You can grant permissions to any of the principals using built-in functions like `give_permission_on_dataset()` and `get_principal_datasets()`. When you grant a permission, you specify:

* Which principal gets the permission
* Which dataset the permission applies to
* What type of permission (read, write, delete, share)

This unified approach means you can grant permissions to:

* Individual [users](./users) for personal access
* [Tenants](./tenants) for organization-wide access
* [Roles](./roles) for team-based access within a tenant

<Accordion title="Principal Model Fields">
  The base Principal model defines what gets stored in the SQL database. The `principals` table contains:

  * `id`: Unique identifier (UUID primary key)
  * `created_at`: Timestamp when created
  * `updated_at`: Timestamp when last modified
  * `type`: Discriminator field for polymorphic inheritance

  Each principal type (User, Tenant, Role) has its own table that references the principals table via foreign key, storing additional fields specific to that type.
</Accordion>

<Accordion title="Permission Storage Schema">
  The permission system links principals to datasets with permissions:

  * `principal_id`: References the principal ([user](./users), [tenant](./tenants), or [role](./roles))
  * `dataset_id`: References the [dataset](./datasets)
  * `permission_id`: References the permission type

  This many-to-many relationship allows flexible permission management across different entity types.
</Accordion>

<Accordion title="Key Functions">
  * `give_permission_on_dataset(principal, dataset_id, permission)`: Writes a single ACL row (or reuses an existing one) so a [user](./users), [tenant](./tenants), or [role](./roles) gains read, write, delete, or share on that dataset. It's the building block used after dataset creation or whenever access is delegated.

  * `get_principal_datasets(principal, permission)`: Queries those ACL entries (and the related dataset records) so you can list every dataset where that same principal holds the requested permission—handy for permission checks or UI listings.
</Accordion>

## Permission Inheritance

The principal system supports hierarchical permission inheritance:

1. **Direct permissions** — Explicitly granted to a specific principal
2. **[Role permissions](./roles)** — Inherited through role memberships
3. **[Tenant permissions](./tenants)** — Inherited through tenant membership

When a [user](./users) tries to access data, the system evaluates their effective permissions by combining all three sources. This allows for flexible access control patterns:

* Grant broad permissions at the [tenant](./tenants) level
* Refine access with [role](./roles)-specific permissions
* Override with direct [user](./users) permissions when needed

## Benefits of the Principal System

* **Unified interface** — Same functions work for all principal types
* **Flexible access control** — Support for individual, team, and organization-level permissions
* **Scalable management** — Easy to add new principal types or modify existing ones
* **Consistent behavior** — All principals follow the same permission rules and patterns

<Columns cols={2}>
  <Card title="Users" icon="user" href="/core-concepts/permissions-system/users">
    Learn about individual users and authentication
  </Card>

  <Card title="Tenants" icon="building" href="/core-concepts/permissions-system/tenants">
    Understand organization-level access control
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
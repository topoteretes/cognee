# ACL

> Access Control List system for permission storage and inheritance in Cognee

# ACL: Permission Storage and Inheritance

The ACL (Access Control List) system stores all permissions and handles permission checking at runtime. ACL entries are stored in the `acls` table, with each row linking a [principal](./principals) to a [dataset](./datasets) with a specific permission.

<Info>**Runtime permission calculation** — The system doesn't store "effective permissions" anywhere—it calculates them on demand by querying ACL entries.</Info>

## How ACL Works

When a [user](./users) tries to access data, the system queries all relevant ACL entries and aggregates the permissions. The permission checking function `get_all_user_permission_datasets()` unions the [user](./users)'s direct permissions with those inherited from their [tenant](./tenants) and [roles](./roles), combining all three sources: direct [user](./users) permissions, [tenant](./tenants)-level permissions, and [role](./roles)-level permissions.

This approach ensures permissions are always current and allows for complex permission inheritance without data duplication.

## ACL Storage Schema

The ACL system uses a simple but powerful schema to store permissions:

<Accordion title="ACL Model Fields">
  The ACL model defines what gets stored in the SQL database. The `acls` table contains:

  * `id`: Unique identifier (UUID primary key)
  * `principal_id`: References the [principal](./principals) ([user](./users), [tenant](./tenants), or [role](./roles))
  * `dataset_id`: References the [dataset](./datasets)
  * `permission_id`: References the permission type
  * `created_at`: Timestamp when created
  * `updated_at`: Timestamp when last modified
</Accordion>

<Accordion title="Permission Checking Functions">
  * `get_all_user_permission_datasets(user, permission)`: Queries ACL entries and returns [datasets](./datasets) the [user](./users) can access
  * `give_permission_on_dataset(principal, dataset_id, permission)`: Creates or updates ACL entries
</Accordion>

## Permission Resolution Order

The system evaluates permissions in a specific order:

1. **Direct [user](./users) permissions** — Explicitly granted to the [user](./users)
2. **[Role](./roles) permissions** — Inherited through the [user](./users)'s role memberships
3. **[Tenant](./tenants) permissions** — Inherited through the [user](./users)'s tenant membership

This order allows for flexible permission management where more specific permissions can override broader ones.

## ACL Operations

The ACL system supports several key operations:

* **Grant permissions** — Add new ACL entries to grant access
* **Revoke permissions** — Remove ACL entries to revoke access
* **Check permissions** — Query ACL entries to determine access
* **List permissions** — Get all permissions for a principal or dataset

## Permission Inheritance

The ACL system implements a three-tier inheritance model:

* **User level** — Direct permissions granted to individual users
* **Role level** — Permissions granted to roles, inherited by role members
* **Tenant level** — Permissions granted to tenants, inherited by all tenant members

Users receive the union of all permissions from these three sources, giving them the most permissive access available.

## Performance Considerations

The ACL system is designed for performance:

* **Indexed queries** — Database indexes on principal\_id, dataset\_id, and permission\_id
* **Efficient lookups** — Single query to get all permissions for a user
* **Caching opportunities** — Permission results can be cached for frequently accessed datasets
* **Batch operations** — Support for granting/revoking multiple permissions at once

## Security Features

The ACL system includes several security features:

* **Immutable ownership** — Dataset ownership cannot be changed
* **Permission validation** — All permission checks go through the ACL system
* **Audit trail** — All permission changes are logged with timestamps
* **Isolation** — Users can only access datasets they have permissions for

## Troubleshooting

Common ACL-related issues and solutions:

* **Permission denied** — Check if user has required permission on the dataset
* **Missing permissions** — Verify ACL entries exist for the principal and dataset
* **Inheritance issues** — Check role and tenant memberships
* **Performance problems** — Review database indexes and query patterns

<Columns cols={2}>
  <Card title="Snippets" icon="code" href="/guides/permission-snippets">
    See practical snippets of ACL operations
  </Card>

  <Card title="Setup Configuration" icon="settings" href="/setup-configuration/permissions">
    Learn how to configure ACL and multi-tenant mode
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
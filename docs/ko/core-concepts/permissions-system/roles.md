# Roles

> Role-based permissions within tenants for granular access control

# Roles

A role is a group of [users](./users) within a [tenant](./tenants). [Roles](./roles) can be granted permissions on [datasets](./datasets), which apply to their members. This enables fine-grained access control within organizations and makes it easier to manage permissions for different teams.

<Info>**Role-based permissions** — When a role is granted a permission on a dataset, all users assigned to that role inherit that permission.</Info>

## Role Concept

[Roles](./roles) are created by [tenant](./tenants) owners and are scoped to that specific [tenant](./tenants). The role belongs to exactly one [tenant](./tenants) as soon as it's created. Because of that foreign-key link, a role can't be moved or shared with another [tenant](./tenants); you would need to create a new role under the other [tenant](./tenants) instead.

[Users](./users) can be assigned to multiple [roles](./roles) within their [tenant](./tenants), and [roles](./roles) can contain multiple [users](./users). This many-to-many relationship allows flexible permission management across teams.

## Role-Based Permissions

When a [role](./roles) is granted a permission on a [dataset](./datasets), all [users](./users) assigned to that role inherit that permission. [Users](./users) receive the union of their direct permissions, [tenant](./tenants)-level permissions, and [role](./roles)-level permissions.

[Roles](./roles) allow you to create permission groups like "editors" or "viewers" within a [tenant](./tenants), making it easier to manage access for different teams without granting permissions to individual [users](./users).

<Accordion title="Role Model Fields">
  The Role model defines what gets stored in the SQL database. The `roles` table contains:

  * `id`: Unique identifier (UUID primary key, references principals.id)
  * `name`: Human-readable name (unique within tenant)
  * `tenant_id`: ID of the tenant this role belongs to (required)
</Accordion>

<Accordion title="Role Creation">
  * `create_role(role_name, owner_id)`: Creates a new role (tenant owner only)
  * `add_user_to_role(user_id, role_id, owner_id)`: Assigns a user to a role (tenant owner only)
</Accordion>

<Accordion title="Limitations">
  * Roles are tenant-scoped and cannot cross tenants
  * API endpoints for role management
</Accordion>

## Role Management

Tenant owners can:

* Create roles within their tenant
* Assign users to roles
* Remove users from roles
* Grant permissions to roles
* Delete roles (when no longer needed)

## Common Role Patterns

Roles are typically organized around job functions or responsibilities:

* **Editors** — Can modify content and run cognify operations
* **Viewers** — Can only read and search data
* **Administrators** — Can manage permissions and users
* **Project Managers** — Can access specific project datasets
* **Reviewers** — Can read and provide feedback on content

## Permission Inheritance Hierarchy

Users receive permissions through a three-level hierarchy:

1. **Direct permissions** — Explicitly granted to the user
2. **Role permissions** — Inherited through role memberships
3. **Tenant permissions** — Inherited through tenant membership

The system calculates effective permissions by combining all three sources, giving users the most permissive access available to them.

## Best Practices

* **Create meaningful role names** — Use descriptive names that reflect the role's purpose
* **Keep roles focused** — Each role should have a clear, specific purpose
* **Regular role reviews** — Periodically review and update role assignments
* **Document role purposes** — Keep clear documentation of what each role is for
* **Principle of least privilege** — Grant only the minimum permissions necessary

## Role vs Tenant Permissions

* **Tenant permissions** — Broad, organization-wide access
* **Role permissions** — Specific, team-based access within the tenant
* **Direct permissions** — Individual, user-specific access

This three-tier system allows for flexible and scalable permission management.

<Columns cols={2}>
  <Card title="ACL" icon="shield" href="/core-concepts/permissions-system/acl">
    Learn how permissions are stored and checked
  </Card>

  <Card title="Snippets" icon="code" href="/guides/permission-snippets">
    See practical snippets of role-based permissions
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
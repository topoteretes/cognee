# Users

> Individual users and authentication in Cognee's permission system

# Users

Users are the most common type of principal and the primary way people access the system. They authenticate through email and password and can own datasets, be granted permissions on others, and belong to at most one tenant.

<Info>**Default user behavior** — When no user is specified, Cognee uses a default user with email "[default\_user@example.com](mailto:default_user@example.com)" for development and testing.</Info>

## User Authentication

Users authenticate through email and password. When no user is specified, Cognee uses a default user with email "[default\_user@example.com](mailto:default_user@example.com)" for development and testing. A user without a tenant can still use the system but operates in isolation.

## User Management

Users can:

* Own [datasets](./datasets) and be granted permissions on others
* Belong to at most one [tenant](./tenants)
* Have direct permissions on [datasets](./datasets)
* Inherit permissions from their [tenant](./tenants) and [roles](./roles)

<Accordion title="User Model Fields">
  The User model defines what gets stored in the SQL database. The `users` table contains:

  * `id`: Unique identifier (UUID primary key, references principals.id)
  * `email`: User's email address (unique)
  * `hashed_password`: Encrypted password
  * `tenant_id`: ID of the [tenant](./tenants) the user belongs to (nullable)
  * `is_active`: Whether the user account is active
  * `is_verified`: Whether the user's email is verified
  * `is_superuser`: Whether the user has superuser privileges
</Accordion>

<Accordion title="User Creation">
  * `create_user(email, password, tenant_id=None, is_superuser=False)`: Creates a new user with specified credentials
  * Default user behavior: System creates "[default\_user@example.com](mailto:default_user@example.com)" if no user exists
</Accordion>

<Accordion title="Environment Variables">
  * `DEFAULT_USER_EMAIL`: Override default user email (default: "[default\_user@example.com](mailto:default_user@example.com)")
  * `DEFAULT_USER_PASSWORD`: Override default user password (default: "default\_password")
  * `REQUIRE_AUTHENTICATION`: Enforce authentication on HTTP endpoints (default: "false")
  * `FASTAPI_USERS_RESET_PASSWORD_TOKEN_SECRET`: Secret for password reset tokens
  * `FASTAPI_USERS_VERIFICATION_TOKEN_SECRET`: Secret for email verification tokens
</Accordion>

<Accordion title="Limitations">
  * A user can belong to at most one tenant
  * Users without a tenant exist but are isolated
  * API endpoints for user management and authentication
</Accordion>

## User Permissions

Users can receive permissions in three ways:

1. **Direct permissions** — Explicitly granted to the user
2. **[Tenant permissions](./tenants)** — Inherited through [tenant](./tenants) membership
3. **[Role permissions](./roles)** — Inherited through [role](./roles) memberships

The system calculates effective permissions by combining all three sources, giving users the union of their direct permissions, [tenant](./tenants)-level permissions, and [role](./roles)-level permissions.

## User Isolation

When `ENABLE_BACKEND_ACCESS_CONTROL=true`, each user's data is completely isolated:

* **Database routing is automatic** — Kùzu (graph) and LanceDB (vector) are configured per request via context variables
* **Filesystem isolation** — Each user gets their own database directory
* **No cross-user access** — Users can only access [datasets](./datasets) they have explicit permissions for

## Superuser Privileges

Users with `is_superuser=True` have additional privileges:

* Can manage other [users](./users), [tenants](./tenants), and [roles](./roles)
* Can access all [datasets](./datasets) regardless of permissions
* Can perform administrative operations

<Warning>**Production Security** — Superuser privileges should be carefully managed in production environments.</Warning>

<Columns cols={2}>
  <Card title="Tenants" icon="building" href="/core-concepts/permissions-system/tenants">
    Learn about organization-level access control
  </Card>

  <Card title="Roles" icon="users" href="/core-concepts/permissions-system/roles">
    Understand role-based permissions within tenants
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
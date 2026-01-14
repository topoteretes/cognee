# Permission Snippets

> Practical code snippets and scenarios for Cognee's permission system

This guide provides practical code snippets demonstrating the permission system in action. These snippets show how to create users, tenants, roles, and datasets, and how to manage permissions effectively.

<Info>**Complete snippets** — All code snippets are complete and runnable, showing the full workflow from setup to permission management.</Info>

<Accordion title="Creating a User">
  [Users](../core-concepts/permissions-system/users) are the foundation of the permission system. Here's how to create a new [user](../core-concepts/permissions-system/users):

  ```python  theme={null}
  from cognee.modules.users.methods import create_user

  user = await create_user(
      email="alice@company.com",
      password="password123",
      is_superuser=True
  )
  ```
</Accordion>

<Accordion title="Creating a Tenant">
  [Tenants](../core-concepts/permissions-system/tenants) group [users](../core-concepts/permissions-system/users) together and can receive permissions. Create a [tenant](../core-concepts/permissions-system/tenants) with an owner:

  ```python  theme={null}
  from cognee.modules.users.tenants.methods import create_tenant

  # Assuming user is already created
  await create_tenant("acme_corp", user.id)
  ```
</Accordion>

<Accordion title="Adding Users to a Tenant">
  Add existing [users](../core-concepts/permissions-system/users) to a [tenant](../core-concepts/permissions-system/tenants). Only the [tenant](../core-concepts/permissions-system/tenants) owner can add [users](../core-concepts/permissions-system/users):

  ```python  theme={null}
  from cognee.modules.users.tenants.methods import add_user_to_tenant

  # Assuming user2, tenant_id, and owner_id are already defined
  await add_user_to_tenant(user2.id, tenant_id, owner_id)
  ```
</Accordion>

<Accordion title="Creating a Role">
  [Roles](../core-concepts/permissions-system/roles) provide permission groups within a [tenant](../core-concepts/permissions-system/tenants). Create a [role](../core-concepts/permissions-system/roles) for the [tenant](../core-concepts/permissions-system/tenants):

  ```python  theme={null}
  from cognee.modules.users.roles.methods import create_role

  # Assuming owner_id is the tenant owner
  await create_role("editor", owner_id)
  ```
</Accordion>

<Accordion title="Creating a Dataset">
  Datasets are the core data containers. Create a dataset with automatic permissions for the creator:

  ```python  theme={null}
  from cognee.modules.data.methods import create_authorized_dataset

  # Assuming user is already created
  dataset = await create_authorized_dataset("project_docs", user)
  ```
</Accordion>

<Accordion title="Granting Read Permission">
  Grant specific permissions to principals. Give read access to a user:

  ```python  theme={null}
  from cognee.modules.users.permissions.methods import give_permission_on_dataset

  # Assuming user2 and dataset are already created
  await give_permission_on_dataset(user2, dataset.id, "read")
  ```
</Accordion>

<Accordion title="Granting Multiple Permissions">
  Grant different permission types to the same principal. Give comprehensive access:

  ```python  theme={null}
  from cognee.modules.users.permissions.methods import give_permission_on_dataset

  # Assuming user2 and dataset are already created
  await give_permission_on_dataset(user2, dataset.id, "read")
  await give_permission_on_dataset(user2, dataset.id, "write")
  await give_permission_on_dataset(user2, dataset.id, "delete")
  ```
</Accordion>

<Accordion title="Checking User Permissions">
  Query what datasets a user can access. Check permissions by type:

  ```python  theme={null}
  from cognee.modules.users.permissions.methods import get_all_user_permission_datasets

  # Assuming user is already created
  # Get all datasets user can read
  readable_datasets = await get_all_user_permission_datasets(user, "read")

  # Get all datasets user can write
  writable_datasets = await get_all_user_permission_datasets(user, "write")
  ```
</Accordion>

<Accordion title="Complete Permission Setup">
  Set up a complete permission scenario from scratch. This example shows the full workflow:

  ```python  theme={null}
  from cognee.modules.users.methods import create_user, get_user
  from cognee.modules.users.tenants.methods import create_tenant, add_user_to_tenant
  from cognee.modules.data.methods import create_authorized_dataset
  from cognee.modules.users.permissions.methods import give_permission_on_dataset

  # 1. Create users
  user1 = await create_user("alice@company.com", "password123", is_superuser=True)
  user2 = await create_user("bob@company.com", "password456")

  # 2. Create tenant and add users
  await create_tenant("acme_corp", user1.id)
  # Refresh user1 to get tenant_id
  user1 = await get_user(user1.id)
  await add_user_to_tenant(user2.id, user1.tenant_id, user1.id)

  # 3. Create dataset
  dataset = await create_authorized_dataset("confidential_docs", user1)

  # 4. Grant different permissions
  await give_permission_on_dataset(user2, dataset.id, "read")  # Read-only access
  ```
</Accordion>

<Accordion title="Permission Inheritance Example">
  Demonstrate how permissions flow through the hierarchy. Show tenant and role inheritance:

  ```python  theme={null}
  from cognee.modules.users.permissions.methods import give_permission_on_dataset

  # Assuming tenant, role, and dataset are already created
  # Grant permission to tenant (all users inherit)
  await give_permission_on_dataset(tenant, dataset.id, "read")

  # Grant permission to role (role members inherit)
  await give_permission_on_dataset(role, dataset.id, "write")

  # User gets both: read (from tenant) + write (from role)
  ```
</Accordion>

<Accordion title="Multi-tenant Organization Setup">
  Create organization with multiple teams:

  ```python  theme={null}
  # Create organization with multiple teams
  # 1. Create tenant
  tenant = await create_tenant("tech_company", admin_user.id)

  # 2. Create roles for different teams
  dev_role = await create_role("developers", admin_user.id)
  qa_role = await create_role("qa_team", admin_user.id)
  pm_role = await create_role("product_managers", admin_user.id)

  # 3. Create datasets for different projects
  frontend_dataset = await create_authorized_dataset("frontend_docs", admin_user)
  backend_dataset = await create_authorized_dataset("backend_docs", admin_user)
  qa_dataset = await create_authorized_dataset("qa_docs", admin_user)

  # 4. Grant role-based permissions
  await give_permission_on_dataset(dev_role, frontend_dataset.id, "write")
  await give_permission_on_dataset(dev_role, backend_dataset.id, "write")
  await give_permission_on_dataset(qa_role, qa_dataset.id, "write")
  await give_permission_on_dataset(pm_role, frontend_dataset.id, "read")
  await give_permission_on_dataset(pm_role, backend_dataset.id, "read")
  ```
</Accordion>

<Accordion title="Temporary Access Management">
  Grant temporary access to external contractor:

  ```python  theme={null}
  # Grant temporary access to external contractor
  contractor = await create_user("contractor@external.com", "temp_password")

  # Grant read access to specific dataset
  await give_permission_on_dataset(contractor, project_dataset.id, "read")

  # Later, revoke access by removing the permission
  # (This would require a revoke_permission function)
  ```
</Accordion>

<Accordion title="Cross-team Collaboration">
  Allow teams to collaborate on shared datasets:

  ```python  theme={null}
  # Allow teams to collaborate on shared datasets
  shared_dataset = await create_authorized_dataset("shared_research", admin_user)

  # Grant different levels of access to different teams
  await give_permission_on_dataset(dev_role, shared_dataset.id, "read")
  await give_permission_on_dataset(research_role, shared_dataset.id, "write")
  await give_permission_on_dataset(management_role, shared_dataset.id, "read")
  ```
</Accordion>

<Accordion title="Best Practices">
  Follow these best practices for permission management:

  * **Start simple** — Begin with basic user and dataset creation
  * **Use roles for teams** — Create roles for different job functions
  * **Grant tenant permissions** — Use tenant-level permissions for organization-wide access
  * **Regular audits** — Periodically review and update permissions
  * **Document access patterns** — Keep clear records of who has access to what
  * **Test permission changes** — Verify permissions work as expected after changes
</Accordion>

<Columns cols={2}>
  <Card title="Setup Configuration" icon="settings" href="/setup-configuration/permissions">
    Learn how to configure the permission system
  </Card>

  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Explore permission system API endpoints
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt
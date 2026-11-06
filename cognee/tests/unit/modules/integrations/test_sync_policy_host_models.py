"""The sync policy store imports next to a host's own credential model (SDK-1434).

Cloud's control plane defines its own ``IntegrationCredential`` on cognee's ``Base``
and the same table name. Importing the store must not load cognee's, or SQLAlchemy
refuses the second definition and Cloud cannot use the store. A fresh interpreter,
because ``Base.metadata`` is process-global.
"""

import subprocess
import sys
import textwrap

HOST_SCRIPT = textwrap.dedent(
    """
    import sys
    import uuid

    from sqlalchemy import Uuid
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.orm import Mapped, mapped_column
    from sqlalchemy.schema import CreateTable

    from cognee.infrastructure.databases.relational import Base


    class HostIntegrationCredential(Base):
        __tablename__ = "integration_credentials"
        id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)


    import cognee.modules.integrations.sync_policy as sync_policy
    from cognee.modules.integrations.sync_policy.models import IntegrationSyncPolicy

    assert "cognee.modules.integrations.models.IntegrationCredential" not in sys.modules
    ddl = str(CreateTable(IntegrationSyncPolicy.__table__).compile(dialect=postgresql.dialect()))
    assert "REFERENCES integration_credentials (id)" in ddl, ddl
    print("ok")
    """
)


REGISTRATION_SCRIPT = textwrap.dedent(
    """
    import cognee
    from cognee.infrastructure.databases.relational import Base

    # The import is the test.
    import cognee.modules.agents.delete_agents_of_user

    assert "integration_sync_policies" not in Base.metadata.tables
    print("ok")
    """
)


def test_importing_the_user_deletion_registers_no_policy_table():
    """The policy table's foreign key needs the host's credential table, so a module that
    only deletes rows must not register it: a later create_all would fail without one."""
    result = subprocess.run(
        [sys.executable, "-c", REGISTRATION_SCRIPT], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")


def test_the_store_imports_beside_a_host_defined_credential_table():
    result = subprocess.run(
        [sys.executable, "-c", HOST_SCRIPT], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")

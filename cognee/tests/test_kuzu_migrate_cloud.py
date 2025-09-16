import unittest
import shutil
import os
import subprocess
import sys
import pathlib
import kuzu

import cognee
from cognee.infrastructure.databases.graph.kuzu.kuzu_migrate import kuzu_migration
from cognee.infrastructure.files.storage import get_file_storage
from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter


# --- Test Configuration ---
# We will test migrating from v0.11.0 to v0.12.0
OLD_KUZU_VERSION = "0.9.0"
NEW_KUZU_VERSION = kuzu.__version__
TEST_NODE_NAME = "Cognee"


class TestKuzuMigrationCloud(unittest.IsolatedAsyncioTestCase):
    """
    Integration test suite for the Kuzu clouddatabase migration script.
    This test will create real virtual environments and databases in a temporary directory.
    """

    async def asyncSetUp(self):
        """
        Set up a temporary directory and create a sample Kuzu database
        with a specific older version before each test.
        """
        try:
            # Create a temporary directory for all test artifacts
            local_test_directory_path = str(
                pathlib.Path(
                    os.path.join(
                        pathlib.Path(__file__).parent, ".cognee_system/test_kuzu_migrate_cloud"
                    )
                ).resolve()
            )
            self.test_dir = local_test_directory_path
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
            os.makedirs(self.test_dir)

            # Define paths for the old and new databases within the temp directory
            self.old_db_name = "old_db"
            self.new_db_name = "new_db"
            self.old_db_path_local = os.path.join(self.test_dir, self.old_db_name)
            self.new_db_path_local = os.path.join(self.test_dir, self.new_db_name)

            print(f"\n--- Setting up test: {self._testMethodName} ---")
            print(f"Creating sample database with Kuzu v{OLD_KUZU_VERSION}...")
            # Create the initial old database required for the test
            self.old_py = self._ensure_env(OLD_KUZU_VERSION, self.test_dir)
            self.new_py = self._ensure_env(NEW_KUZU_VERSION, self.test_dir)
            self._run_migration_step(
                self.old_py,
                snippet=f"""
import kuzu
db = kuzu.Database(r"{self.old_db_path_local}")
conn = kuzu.Connection(db)
conn.execute('CREATE NODE TABLE TestNode(name STRING, PRIMARY KEY(name))')
conn.execute("CREATE (t:TestNode {{name: '{TEST_NODE_NAME}'}})")
print("DB created successfully.")
""",
            )
            print("Setup test local database complete.")

            # Setup test cloud database
            bucket_name = os.getenv("STORAGE_BUCKET_NAME")

            cognee_directory_path = f"s3://{bucket_name}/test_kuzu_migrate_cloud/system/databases"
            cognee.config.system_root_directory(cognee_directory_path)

            # Push the old database to the cloud
            self.file_storage = get_file_storage(cognee_directory_path)
            await self.file_storage.remove_all()  # Clean up any existing databases
            self.old_db_path = os.path.join(cognee_directory_path, self.old_db_name)
            self.new_db_path = os.path.join(cognee_directory_path, self.new_db_name)
            await self.file_storage.storage.push_to_cloud(self.old_db_name, self.old_db_path_local)
            print("Setup test cloud database complete.")
            print(f"\n--- Setup complete for: {self._testMethodName} ---")

        except Exception as e:
            print(f"Error setting up test: {e}")
            await self.file_storage.remove_all()
            shutil.rmtree(self.test_dir)

    async def asyncTearDown(self):
        """
        Clean up the temporary directory and cloud test database after each test.
        """
        await self.file_storage.remove_all()
        shutil.rmtree(self.test_dir)
        print(f"--- Teardown complete for: {self._testMethodName} ---")

    def _ensure_env(self, version: str, export_dir) -> str:
        """
        Create (if needed) a venv at .kuzu_envs/{version} and install kuzu=={version}.
        Returns the path to the venv's python executable.
        """
        # Use temp directory to create venv
        kuzu_envs_dir = os.path.join(export_dir, ".kuzu_envs")

        # venv base under the script directory
        base = os.path.join(kuzu_envs_dir, version)
        py_bin = os.path.join(base, "bin", "python")
        # If environment already exists clean it
        if os.path.isfile(py_bin):
            shutil.rmtree(base)

        print(f"→ Setting up venv for Kuzu {version}...", file=sys.stderr)
        # Create venv
        subprocess.run([sys.executable, "-m", "venv", base], check=True)
        # Install the specific Kùzu version
        subprocess.run([py_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([py_bin, "-m", "pip", "install", f"kuzu=={version}"], check=True)
        print(f"→ Venv for Kuzu {version} setup complete.")
        return py_bin

    def _run_migration_step(self, python_exe: str, snippet: str):
        """
        Uses the given python_exe to execute a short snippet that
        connects to the Kùzu database and runs a Cypher command.
        """

        proc = subprocess.run([python_exe, "-c", snippet], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[ERROR] {snippet} failed:\n{proc.stderr}", file=sys.stderr)
            shutil.rmtree(self.test_dir)
            sys.exit(proc.returncode)
        return proc.stdout.strip()

    def _verify_db_data(self, python_exe: str, db_path: str, kuzu_version: str):
        """
        Connects to a Kuzu database and verifies that the test data exists.
        Fails the test if the data is not found.
        """
        print(f"Verifying data in {db_path} using Kuzu v{kuzu_version}...")
        verify_script = f"""
import kuzu
db = kuzu.Database('{db_path}')
conn = kuzu.Connection(db)
result = conn.execute('MATCH (t:TestNode) RETURN t.name').get_next()
# result is a list, e.g., ['Cognee'], so we access the first element
print(result[0])
"""
        output = self._run_migration_step(python_exe, verify_script)
        self.assertEqual(
            output, TEST_NODE_NAME, "Data verification failed: Node name does not match."
        )
        print("Data verification successful.")

    async def test_successful_migration(self):
        """
        Tests a standard migration without overwrite or delete flags.
        Expects the new database to be created at --new-db and the old one to remain untouched.
        """
        kuzu_migration(
            new_db=self.new_db_path,
            old_db=self.old_db_path,
            new_version=NEW_KUZU_VERSION,
            old_version=OLD_KUZU_VERSION,
        )

        # Assert that both the original and new databases exist
        old_db_exists = await self.file_storage.file_exists(self.old_db_name)
        self.assertTrue(
            old_db_exists,
            "Old database should still exist.",
        )
        new_db_exists = await self.file_storage.file_exists(self.new_db_name)
        self.assertTrue(
            new_db_exists,
            "New database should have been created.",
        )

    async def test_migration_with_overwrite(self):
        """
        Tests migration with the --overwrite flag.
        Expects the new database to replace the old one, and the original old database
        to be backed up with an '_old' suffix.
        """
        kuzu_migration(
            new_db=self.new_db_path,
            old_db=self.old_db_path,
            new_version=NEW_KUZU_VERSION,
            old_version=OLD_KUZU_VERSION,
            overwrite=True,
        )

        # The intermediate new_db path should be gone (it was moved)
        new_db_exists = await self.file_storage.file_exists(self.new_db_name)
        self.assertFalse(
            new_db_exists,
            "Intermediate new DB should not exist after overwrite.",
        )

        # The new, migrated database should now exist at the original path
        old_db_exists = await self.file_storage.file_exists(self.old_db_name)
        self.assertTrue(
            old_db_exists,
            "Migrated DB should exist at the original path.",
        )

        # Check for the backup directory
        backup_db_name = f"old_db_old_{OLD_KUZU_VERSION.replace('.', '_')}"

        backup_db_exists = await self.file_storage.file_exists(backup_db_name)
        self.assertTrue(
            backup_db_exists,
            "Backup of old database should exist.",
        )

        # Verify the data in the final (overwritten) database
        await self.file_storage.storage.pull_from_cloud(self.old_db_name, self.new_db_path_local)
        self._verify_db_data(self.new_py, self.new_db_path_local, NEW_KUZU_VERSION)

    async def test_migration_with_delete_old(self):
        """
        Tests migration with both --overwrite and --delete-old flags.
        Expects the new database to replace the old one, with no backup created.
        """
        kuzu_migration(
            new_db=self.new_db_path,
            old_db=self.old_db_path,
            new_version=NEW_KUZU_VERSION,
            old_version=OLD_KUZU_VERSION,
            overwrite=True,
            delete_old=True,
        )

        old_db_exists = await self.file_storage.file_exists(self.old_db_name)
        self.assertTrue(
            old_db_exists,
            "Migrated DB should exist at the original path.",
        )

        new_db_exists = await self.file_storage.file_exists(self.new_db_name)
        self.assertFalse(
            new_db_exists,
            "Intermediate new DB should have been moved.",
        )

        # The backup directory should NOT exist
        backup_db_name = f"old_db_old_{OLD_KUZU_VERSION.replace('.', '_')}"

        backup_db_exists = await self.file_storage.file_exists(backup_db_name)
        self.assertFalse(
            backup_db_exists,
            "Backup of old database should NOT exist.",
        )

        # Verify the data in the final database
        await self.file_storage.storage.pull_from_cloud(self.old_db_name, self.new_db_path_local)
        self._verify_db_data(self.new_py, self.new_db_path_local, NEW_KUZU_VERSION)

    async def test_version_auto_detection(self):
        """
        Tests a migration where the old_version is not provided, forcing the script
        to use its automatic version detection logic.
        """
        kuzu_migration(
            new_db=self.new_db_path,
            old_db=self.old_db_path,
            new_version=NEW_KUZU_VERSION,
            old_version=None,  # Omit the old version to test detection
        )

        new_db_exists = await self.file_storage.file_exists(self.new_db_name)
        self.assertTrue(
            new_db_exists,
            "New database should be created after auto-detection.",
        )

        await self.file_storage.storage.pull_from_cloud(self.new_db_name, self.new_db_path_local)
        self._verify_db_data(self.new_py, self.new_db_path_local, NEW_KUZU_VERSION)

    async def test_new_db_already_exists_error(self):
        """
        Tests that the script fails correctly if the --new-db path already exists
        before migration starts.
        """
        # Create a dummy file at the target location to trigger the error
        await self.file_storage.storage.push_to_cloud(self.new_db_name, self.old_db_path_local)
        with self.assertRaisesRegex(
            FileExistsError,
            "File already exists at new database location, remove file or change new database file path to continue",
        ):
            kuzu_migration(
                new_db=self.new_db_path,
                old_db=self.old_db_path,
                new_version=NEW_KUZU_VERSION,
                old_version=OLD_KUZU_VERSION,
            )

    async def test_kuzu_adpter_init_migrate(self):
        KuzuAdapter(self.old_db_path)

        # The new, migrated database should now exist at the original path
        old_db_exists = await self.file_storage.file_exists(self.old_db_name)
        self.assertTrue(
            old_db_exists,
            "Migrated DB should exist at the original path.",
        )

        # Check for the backup directory
        backup_db_name = f"old_db_old_{OLD_KUZU_VERSION.replace('.', '_')}"

        backup_db_exists = await self.file_storage.file_exists(backup_db_name)
        self.assertTrue(
            backup_db_exists,
            "Backup of old database should exist.",
        )

        # Verify the data in the final (overwritten) database
        await self.file_storage.storage.pull_from_cloud(self.old_db_name, self.new_db_path_local)
        self._verify_db_data(self.new_py, self.new_db_path_local, NEW_KUZU_VERSION)


if __name__ == "__main__":
    unittest.main()

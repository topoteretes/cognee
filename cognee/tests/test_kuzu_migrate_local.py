import unittest
import pathlib
import shutil
import os
import subprocess
import sys
import kuzu

import cognee
from cognee.infrastructure.databases.graph.kuzu.kuzu_migrate import kuzu_migration

from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter


# --- Test Configuration ---
OLD_KUZU_VERSION = "0.8.2"
NEW_KUZU_VERSION = kuzu.__version__
TEST_NODE_NAME = "Cognee"


class TestKuzuMigrationLocal(unittest.TestCase):
    """
    Integration test suite for the Kuzu database migration script.
    This test will create real virtual environments and databases in a temporary directory.
    """

    def setUp(self):
        """
        Set up a temporary directory and create a sample Kuzu database
        with a specific older version before each test.
        """
        try:
            # Create a temporary directory for all test artifacts
            cognee_directory_path = str(
                pathlib.Path(
                    os.path.join(
                        pathlib.Path(__file__).parent, ".cognee_system/test_kuzu_migrate_local"
                    )
                ).resolve()
            )

            cognee.config.system_root_directory(cognee_directory_path)
            self.test_dir = cognee_directory_path
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
            os.makedirs(self.test_dir)

            # Define paths for the old and new databases within the temp directory
            self.old_db_path = os.path.join(self.test_dir, "old_db")
            self.new_db_path = os.path.join(self.test_dir, "new_db")

            print(f"\n--- Setting up test: {self._testMethodName} ---")
            print(f"Creating sample database with Kuzu v{OLD_KUZU_VERSION}...")
            # Create the initial old database required for the test
            self.old_py = self._ensure_env(OLD_KUZU_VERSION, self.test_dir)
            self.new_py = self._ensure_env(NEW_KUZU_VERSION, self.test_dir)
            self._run_migration_step(
                self.old_py,
                snippet=f"""
import kuzu
db = kuzu.Database(r"{self.old_db_path}")
conn = kuzu.Connection(db)
conn.execute('CREATE NODE TABLE TestNode(name STRING, PRIMARY KEY(name))')
conn.execute("CREATE (t:TestNode {{name: '{TEST_NODE_NAME}'}})")
print("DB created successfully.")
""",
            )
            print(f"\n--- Setup complete for: {self._testMethodName} ---")
        except Exception as e:
            print(f"Error setting up test: {e}")
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        """
        Clean up the temporary directory and all its contents after each test.
        """
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

    def test_successful_migration(self):
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
        self.assertTrue(os.path.exists(self.old_db_path), "Old database should still exist.")
        self.assertTrue(os.path.exists(self.new_db_path), "New database should have been created.")

    def test_migration_with_overwrite(self):
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
        self.assertFalse(
            os.path.exists(self.new_db_path),
            "Intermediate new DB should not exist after overwrite.",
        )

        # The new, migrated database should now exist at the original path
        self.assertTrue(
            os.path.exists(self.old_db_path), "Migrated DB should exist at the original path."
        )

        # Check for the backup directory
        backup_db_name = f"old_db_old_{OLD_KUZU_VERSION.replace('.', '_')}"
        backup_path = os.path.join(self.test_dir, backup_db_name)
        self.assertTrue(os.path.exists(backup_path), "Backup of old database should exist.")

        # Verify the data in the final (overwritten) database
        self._verify_db_data(self.new_py, self.old_db_path, NEW_KUZU_VERSION)

    def test_migration_with_delete_old(self):
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

        self.assertTrue(
            os.path.exists(self.old_db_path), "Migrated DB should exist at the original path."
        )
        self.assertFalse(
            os.path.exists(self.new_db_path), "Intermediate new DB should have been moved."
        )

        # The backup directory should NOT exist
        backup_db_name = f"old_db_old_{OLD_KUZU_VERSION.replace('.', '_')}"
        backup_path = os.path.join(self.test_dir, backup_db_name)
        self.assertFalse(os.path.exists(backup_path), "Backup of old database should NOT exist.")

        # Verify the data in the final database
        self._verify_db_data(self.new_py, self.old_db_path, NEW_KUZU_VERSION)

    def test_version_auto_detection(self):
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

        self.assertTrue(
            os.path.exists(self.new_db_path), "New database should be created after auto-detection."
        )
        self._verify_db_data(self.new_py, self.new_db_path, NEW_KUZU_VERSION)

    def test_new_db_already_exists_error(self):
        """
        Tests that the script fails correctly if the --new-db path already exists
        before migration starts.
        """
        # Create a dummy file at the target location to trigger the error
        os.makedirs(self.new_db_path)

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

    def test_kuzu_adpter_init_migrate(self):
        KuzuAdapter(self.old_db_path)

        # The new, migrated database should now exist at the original path
        self.assertTrue(
            os.path.exists(self.old_db_path), "Migrated DB should exist at the original path."
        )

        # Check for the backup directory
        backup_db_name = f"old_db_old_{OLD_KUZU_VERSION.replace('.', '_')}"
        backup_path = os.path.join(self.test_dir, backup_db_name)
        self.assertTrue(os.path.exists(backup_path), "Backup of old database should exist.")

        # Verify the data in the final (overwritten) database
        self._verify_db_data(self.new_py, self.old_db_path, NEW_KUZU_VERSION)


if __name__ == "__main__":
    unittest.main()

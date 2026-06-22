import os


def configure_cognee_for_subprocess(cognee):
    data_root_directory = os.getenv("COGNEE_TEST_DATA_ROOT")
    system_root_directory = os.getenv("COGNEE_TEST_SYSTEM_ROOT")

    if data_root_directory:
        cognee.config.data_root_directory(data_root_directory)
    if system_root_directory:
        cognee.config.system_root_directory(system_root_directory)


def get_kuzu_db_path() -> str:
    return os.getenv("COGNEE_TEST_KUZU_DB_PATH", "test.db")

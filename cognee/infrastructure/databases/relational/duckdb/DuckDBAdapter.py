import duckdb

class DuckDBAdapter():
    def __init__(self, db_path: str, db_name: str):
        db_location = db_path + "/" + db_name

        self.db_client = duckdb.connect(db_location)

    def get_datasets(self):
        tables = self.db_client.sql("SELECT DISTINCT schema_name FROM duckdb_tables();").to_df().to_dict("list")

        return list(
            filter(
                lambda table_name: table_name.endswith('staging') is False,
                tables["schema_name"]
            )
        )

    def get_files_metadata(self, dataset_name: str):
        return self.db_client.sql(f"SELECT id, name, file_path, extension, mime_type, keywords FROM {dataset_name}.file_metadata;").to_df().to_dict("records")

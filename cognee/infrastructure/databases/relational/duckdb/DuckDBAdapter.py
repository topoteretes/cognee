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


    def load_cognify_data(self, data):
        # Ensure the 'cognify' table exists
        self.db_client.execute("""
            CREATE TABLE IF NOT EXISTS cognify (
                document_id STRING,
                layer_id STRING,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT NULL
            );
        """)

        # Prepare the insert statement
        insert_query = """
            INSERT INTO cognify (document_id, layer_id)
            VALUES (?, ?);
        """

        # Insert each record into the 'cognify' table
        for record in data:
            self.db_client.execute(insert_query, [
                record.get('document_id'),
                record.get('layer_id')
            ])

    def fetch_cognify_data(self, excluded_document_id: str):
        # SQL command to create the 'cognify' table with the specified columns
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS cognify (
            document_id STRING,
            layer_id STRING,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT NULL 
        );
        """
        # Execute the SQL command to create the table
        self.db_client.execute(create_table_sql)

        # SQL command to select data from the 'cognify' table
        select_data_sql = f"SELECT document_id, layer_id, created_at, updated_at FROM cognify WHERE document_id != '{excluded_document_id}';"

        # Execute the query and return the results as a list of dictionaries
        return self.db_client.sql(select_data_sql).to_df().to_dict("records")


    def delete_cognify_data(self):
        # SQL command to create the 'cognify' table with the specified columns
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS cognify (
            document_id STRING,
            layer_id STRING,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT NULL 
        );
        """
        # Execute the SQL command to create the table
        self.db_client.execute(create_table_sql)

        # SQL command to select data from the 'cognify' table
        select_data_sql = "DELETE FROM cognify;"

        # Execute the query and return the results as a list of dictionaries
        return self.db_client.sql(select_data_sql).to_df().to_dict("records")
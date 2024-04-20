import duckdb

class DuckDBAdapter():
    def __init__(self, db_path: str, db_name: str):
        db_location = db_path + "/" + db_name

        self.get_connection = lambda: duckdb.connect(db_location)

    def get_datasets(self):
        with self.get_connection() as connection:
            tables = connection.sql("SELECT DISTINCT schema_name FROM duckdb_tables();").to_df().to_dict("list")

        return list(
            filter(
                lambda table_name: table_name.endswith("staging") is False,
                tables["schema_name"]
            )
        )

    def get_files_metadata(self, dataset_name: str):
        with self.get_connection() as connection:
            return connection.sql(f"SELECT id, name, file_path, extension, mime_type, keywords FROM {dataset_name}.file_metadata;").to_df().to_dict("records")


    def load_cognify_data(self, data):
        with self.get_connection() as connection:
            # Ensure the "cognify" table exists
            connection.execute("""
                CREATE TABLE IF NOT EXISTS cognify (
                    document_id STRING,
                    layer_id STRING,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    document_id_target STRING NULL
                );
            """)

        # Prepare the insert statement
        insert_query = """
            INSERT INTO cognify (document_id, layer_id)
            VALUES (?, ?);
        """

        # Insert each record into the "cognify" table
        for record in data:
            with self.get_connection() as connection:
                connection.execute(insert_query, [
                    record.get("document_id"),
                    record.get("layer_id")
                ])

    def fetch_cognify_data(self, excluded_document_id: str):
        # SQL command to create the "cognify" table with the specified columns
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS cognify (
            document_id STRING,
            layer_id STRING,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT NULL,
            processed BOOLEAN DEFAULT FALSE,
            document_id_target STRING NULL
        );
        """
        with self.get_connection() as connection:
            # Execute the SQL command to create the table
            connection.execute(create_table_sql)

        # SQL command to select data from the "cognify" table
        select_data_sql = f"SELECT document_id, layer_id, created_at, updated_at, processed FROM cognify WHERE document_id != '{excluded_document_id}' AND processed = FALSE;"

        with self.get_connection() as connection:
            # Execute the query and fetch the results
            records = connection.sql(select_data_sql).to_df().to_dict("records")

        # If records are fetched, update the "processed" column to "True"
        if records:
            # Fetching document_ids from the records to update the "processed" column
            document_ids = tuple(record["document_id"] for record in records)
            # SQL command to update the "processed" column to "True" for fetched records
            update_data_sql = f"UPDATE cognify SET processed = TRUE WHERE document_id IN {document_ids};"

            with self.get_connection() as connection:
                # Execute the update query
                connection.execute(update_data_sql)

        # Return the fetched records
        return records


    def delete_cognify_data(self):
        # SQL command to create the "cognify" table with the specified columns
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS cognify (
            document_id STRING,
            layer_id STRING,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT NULL,
            processed BOOLEAN DEFAULT FALSE,
            document_id_target STRING NULL
        );
        """

        with self.get_connection() as connection:
            # Execute the SQL command to create the table
            connection.execute(create_table_sql)

        with self.get_connection() as connection:
            # SQL command to select data from the "cognify" table
            select_data_sql = "DELETE FROM cognify;"
            connection.sql(select_data_sql)
            drop_data_sql = "DROP TABLE cognify;"
            connection.sql(drop_data_sql)

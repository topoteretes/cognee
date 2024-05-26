import duckdb
import os
class DuckDBAdapter():
    def __init__(self, db_path: str, db_name: str):

        db_location = os.path.abspath(os.path.join(db_path, db_name))

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

    def create_table(self, table_name: str, table_config: list[dict]):
        fields_query_parts = []

        for table_config_item in table_config:
            fields_query_parts.append(f"{table_config_item['name']} {table_config_item['type']}")

        with self.get_connection() as connection:
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(fields_query_parts)});"
            connection.execute(query)

    def delete_table(self, table_name: str):
        with self.get_connection() as connection:
            query = f"DROP TABLE IF EXISTS {table_name};"
            connection.execute(query)

    def insert_data(self, table_name: str, data: list[dict]):
        def get_values(data_entry: list):
            return ", ".join([f"'{value}'" if isinstance(value, str) else value for value in data_entry])
      
        columns = ", ".join(data[0].keys())
        values = ", ".join([f"({get_values(data_entry.values())})" for data_entry in data])

        with self.get_connection() as connection:
            query = f"INSERT INTO {table_name} ({columns}) VALUES {values};"
            connection.execute(query)

    def get_data(self, table_name: str, filters: dict = None):
        with self.get_connection() as connection:
            def get_values(values: list):
                return ", ".join([f"'{value}'" for value in values])

            def get_filters(filters: dict):
                return " AND ".join([
                    f"{key} IN ({get_values(value)})" if isinstance(value, list)
                    else f"{key} = '{value}'" for (key, value) in filters.items()
                ])

            query = f"SELECT * FROM {table_name}" + (";" if filters is None else f" WHERE {get_filters(filters)};")
            results = connection.sql(query).to_df().to_dict("records")

            return {
                result["data_id"]: result["status"] for result in results
            }

    def execute_query(self, query):
        with self.get_connection() as connection:
            return connection.sql(query).to_df().to_dict("records")

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

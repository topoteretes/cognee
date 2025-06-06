from .supported_databases import supported_databases


def use_vector_adapter(vector_db_name, vector_db_adapter):
    supported_databases[vector_db_name] = vector_db_adapter

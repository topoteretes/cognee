"""Op-codes exchanged between the Kuzu subprocess worker and its main-side
proxy. Pure stdlib.
"""

from __future__ import annotations

OP_OPEN_DATABASE = 1           # kwargs: db_path, buffer_pool_size, max_num_threads, max_db_size
OP_DB_INIT = 2                 # handle_id = database
OP_DB_CLOSE = 3                # handle_id = database

OP_OPEN_CONNECTION = 10        # args = (database_handle_id,)
OP_CONN_CLOSE = 11             # handle_id = connection

# Execute a query and return fully-materialized rows (preferred path).
# args: (query, params_dict_or_None)
OP_CONN_EXECUTE_FETCH_ALL = 12

# Install the JSON extension via a throwaway database (handled end-to-end in
# the worker). args: (buffer_pool_size,)
OP_INSTALL_JSON = 20

# Load an extension on an existing connection. args: (extension_name,)
OP_LOAD_EXTENSION = 21

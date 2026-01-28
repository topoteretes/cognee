from cognee.api.v1.config.config import config

config.set_vector_db_config({
    "vector_db_provider": "lancedb",  # valid
    "vector_db_urll": "/tmp/db",  # typo: invalid key
})

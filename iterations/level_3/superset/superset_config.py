import os

FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,
}

ENABLE_PROXY_FIX = True
SECRET_KEY = "YOUR_OWN_RANDOM_GENERATED_STRING"  # Make sure to generate and use your own secret key

# PostgreSQL Database credentials
POSTGRES_USER = os.getenv('POSTGRES_USER', 'bla')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'bla')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'bubu')

# Constructing the SQLAlchemy PostgreSQL URI
SQLALCHEMY_DATABASE_URI = (
    f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@'
    f'{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
)

DATABASE_CONNECTIONS = [
    {
        'database_name': 'my_postgres',
        'sqlalchemy_uri':     f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@'
    f'{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}',
    },

    # Add more database connections as needed
]
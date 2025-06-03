import os
from starlette.config import Config
from authlib.integrations.starlette_client import OAuth

config = Config(".env")

oauth = OAuth(config)

oauth.register(
    "auth0",
    client_id=os.getenv("AUTH0_CLIENT_ID"),
    client_secret=os.getenv("AUTH0_CLIENT_SECRET"),
    server_metadata_url=f"https://{os.getenv('AUTH0_DOMAIN')}/.well-known/openid-configuration",
    client_kwargs={"scope": "openid profile email"},
)

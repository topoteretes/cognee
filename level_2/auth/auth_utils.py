
from cognito.JWTBearer import JWKS, JWTBearer, JWTAuthorizationCredentials

import requests

region = "eu-west-1"
user_pool_id = "" #needed
cognito_url = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/jwks.json"

# Fetch the JWKS using the updated URL
jwks = JWKS.parse_obj(requests.get(cognito_url).json())
print(jwks)

auth = JWTBearer(jwks)


import requests

# Set the Cognito authentication endpoint URL

auth = JWTBearer(jwks)

# Set the user credentials

username = "" #needed
password = "" #needed

# Create the authentication payload
payload = {
    "username": username,
    "password": password
}

# Set the Cognito authentication endpoint URL
# Set the Cognito token endpoint URL
token_endpoint = f"https://your-cognito-domain.auth.{region}.amazoncognito.com/oauth2/token"

# Set the client credentials
client_id = "" #needed
client_secret = ""

import boto3
def authenticate_and_get_token(username: str, password: str,
                               user_pool_id: str, app_client_id: str) -> None:
    client = boto3.client('cognito-idp')

    resp = client.admin_initiate_auth(
        UserPoolId=user_pool_id,
        ClientId=app_client_id,
        AuthFlow='ADMIN_NO_SRP_AUTH',
        AuthParameters={
            "USERNAME": username,
            "PASSWORD": password
        }
    )

    print("Log in success")
    print("Access token:", resp['AuthenticationResult']['AccessToken'])
    print("ID token:", resp['AuthenticationResult']['IdToken'])


authenticate_and_get_token(username, password, user_pool_id, client_id)
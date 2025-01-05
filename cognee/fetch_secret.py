import os
import sys
import boto3
from dotenv import load_dotenv

# Get the directory that contains your script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

environment = os.getenv("AWS_ENV", "dev")


def fetch_secret(secret_name: str, region_name: str, env_file_path: str):
    """Fetch the secret from AWS Secrets Manager and write it to the .env file."""
    print("Initializing session")
    session = boto3.session.Session()
    print("Session initialized")
    client = session.client(service_name="secretsmanager", region_name=region_name)
    print("Client initialized")

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        return f"Error retrieving secret: {e}"

    if "SecretString" in response:
        secret = response["SecretString"]
    else:
        secret = response["SecretBinary"]

    with open(env_file_path, "w") as env_file:
        env_file.write(secret)
        print("Secrets are added to the .env file.")

    if os.path.exists(env_file_path):
        print(f"The .env file is located at: {env_file_path}")
        load_dotenv()
        print("The .env file is loaded.")
    else:
        print(f"The .env file was not found at: {env_file_path}.")


ENV_FILE_PATH = os.path.abspath("../.env")

if os.path.exists(ENV_FILE_PATH):
    # Load default environment variables (.env)
    load_dotenv()
    print("Environment variables are already loaded.")
else:
    fetch_secret(
        f"promethai-{environment}-backend-secretso-promethaijs-dotenv",
        "eu-west-1",
        ENV_FILE_PATH,
    )

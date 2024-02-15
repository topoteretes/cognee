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

# API_ENABLED = os.environ.get("API_ENABLED", "False").lower() == "true"

environment = os.getenv("AWS_ENV", "dev")


def fetch_secret(secret_name, region_name, env_file_path):
    print("Initializing session")
    session = boto3.session.Session()
    print("Session initialized")
    client = session.client(service_name="secretsmanager", region_name=region_name)
    print("Client initialized")

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        return None

    if "SecretString" in response:
        secret = response["SecretString"]
    else:
        secret = response["SecretBinary"]

    if os.path.exists(env_file_path):
        print(f"The .env file is located at: {env_file_path}")

        with open(env_file_path, "w") as env_file:
            env_file.write(secret)
            print("Secrets are added to the .env file.")

        load_dotenv()
        print("The .env file is loaded.")
    else:
        print(f"The .env file was not found at: {env_file_path}.")


ENV_FILE_PATH = os.path.abspath("../.env")

if os.path.exists(ENV_FILE_PATH):
    # Load default environment variables (.env)
    load_dotenv()
    print("Cognee is already running...")
else:
    fetch_secret(
        f"promethai-{environment}-backend-secretso-promethaijs-dotenv",
        "eu-west-1",
        ENV_FILE_PATH,
    )

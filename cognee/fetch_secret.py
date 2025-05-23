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
    """Fetch the secret from AWS Secrets Manager and load it into environment variables (DO NOT write to disk)."""
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
        print("Binary secrets are not supported and cannot be loaded as environment variables.")
        return "Error: SecretBinary type is not supported."

    # Parse each line as KEY=VALUE, set in os.environ
    for line in secret.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()
    print("Secrets loaded into environment variables (not written to disk).")

    # Since we are not writing the file, omit writing and loading from file.
    # Just confirm via env.
    for k in os.environ:
        if k in secret:
            print(f"Set environment variable: {k}")


ENV_FILE_PATH = os.path.abspath("../.env")

if os.path.exists(ENV_FILE_PATH):
    # Load default environment variables (.env)
    load_dotenv()
    print("Environment variables are already loaded from .env file.")
else:
    fetch_secret(
        f"promethai-{environment}-backend-secretso-promethaijs-dotenv",
        "eu-west-1",
        ENV_FILE_PATH,
    )
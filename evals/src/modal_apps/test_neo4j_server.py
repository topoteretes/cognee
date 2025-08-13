import socket
import subprocess
import time
from pathlib import Path

import modal

from modal_apps.modal_image import neo4j_env_dict, neo4j_image

APP_NAME = "test-neo4j-server"


app = modal.App(APP_NAME, secrets=[modal.Secret.from_dotenv()])


@app.function(
    image=neo4j_image,
    timeout=3600,
    cpu=2,
    memory=8192,
)
async def start_neo4j_server():
    """Start and maintain Neo4j server for testing."""
    print("Starting Neo4j server process...")

    # Set initial password
    password = neo4j_env_dict["NEO4J_AUTH"].split("/")[1]
    set_password_command = f"neo4j-admin dbms set-initial-password {password}"
    try:
        subprocess.run(
            f"su-exec neo4j:neo4j {set_password_command}",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ Initial password has been set.")
    except subprocess.CalledProcessError as e:
        if "already been set" in e.stderr:
            print("Password has already been set on a previous run.")
        else:
            print("❌ Failed to set initial password:")
            print(e.stderr)
            raise

    # Start Neo4j server
    neo4j_process = subprocess.Popen(
        "su-exec neo4j:neo4j neo4j console",
        shell=True,
    )

    print("Waiting for Neo4j server to become available on port 7474...")
    while True:
        try:
            with socket.create_connection(("localhost", 7474), timeout=1):
                print("✅ Neo4j server is ready on port 7474.")
                break
        except (socket.timeout, ConnectionRefusedError):
            if neo4j_process.poll() is not None:
                raise RuntimeError("Neo4j process terminated unexpectedly.")
            time.sleep(1)

    # Forward both ports and keep server running within tunnel contexts
    with (
        modal.forward(7474, unencrypted=True) as http_tunnel,
        modal.forward(7687, unencrypted=True) as bolt_tunnel,
    ):
        http_host, http_port = http_tunnel.tcp_socket
        print(f"🌐 Neo4j Browser available at: http://{http_host}:{http_port}")
        print(f"🔑 Username: {neo4j_env_dict['NEO4J_AUTH'].split('/')[0]}")
        print(f"🔑 Password: {password}")

        bolt_host, bolt_port = bolt_tunnel.tcp_socket
        bolt_addr = f"bolt://{bolt_host}:{bolt_port}"
        print(f"🔌 Bolt connection: {bolt_addr}")
        print(f"🔑 Username: {neo4j_env_dict['NEO4J_AUTH'].split('/')[0]}")
        print(f"🔑 Password: {password}")

        print("Neo4j server is running. Press Ctrl+C to stop.")

        # Keep the server running within the tunnel contexts
        try:
            neo4j_process.wait()
        except KeyboardInterrupt:
            print("Shutting down Neo4j server...")
            neo4j_process.terminate()
            neo4j_process.wait()
            print("Neo4j server stopped.")


@app.local_entrypoint()
async def main():
    """Start multiple Neo4j servers for testing."""
    print("🚀 Starting 2 Neo4j servers for testing...")

    # Start two separate containers concurrently
    import asyncio

    await asyncio.gather(start_neo4j_server.remote.aio(), start_neo4j_server.remote.aio())

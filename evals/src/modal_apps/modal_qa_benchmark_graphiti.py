import datetime
import os
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import modal

from modal_apps.modal_image import graphiti_image, neo4j_env_dict, neo4j_image

APP_NAME = "qa-benchmark-graphiti"
VOLUME_NAME = "qa-benchmarks"
BENCHMARK_NAME = "graphiti"
CORPUS_FILE = Path("hotpot_qa_24_corpus.json")
QA_PAIRS_FILE = Path("hotpot_qa_24_qa_pairs.json")


def _create_benchmark_folder(volume_name: str, benchmark_name: str, timestamp: str) -> str:
    """Create benchmark folder structure and return the answers folder path."""
    benchmark_folder = f"/{volume_name}/{benchmark_name}_{timestamp}"
    answers_folder = f"{benchmark_folder}/answers"

    # Create directories if they don't exist
    os.makedirs(answers_folder, exist_ok=True)

    return answers_folder


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, secrets=[modal.Secret.from_dotenv()])


@app.function(
    image=graphiti_image,
    volumes={f"/{VOLUME_NAME}": volume},
    timeout=14400,
    cpu=4,
)
def run_graphiti_benchmark(config_params: dict, dir_suffix: str):
    """Run the Graphiti QA benchmark on Modal."""
    from qa.qa_benchmark_graphiti import GraphitiConfig, QABenchmarkGraphiti

    print("Received benchmark request for Graphiti.")

    # Create benchmark folder structure
    answers_folder = _create_benchmark_folder(VOLUME_NAME, BENCHMARK_NAME, dir_suffix)
    print(f"Created benchmark folder: {answers_folder}")

    config = GraphitiConfig(**config_params)

    benchmark = QABenchmarkGraphiti.from_jsons(
        corpus_file=f"/root/{CORPUS_FILE.name}",
        qa_pairs_file=f"/root/{QA_PAIRS_FILE.name}",
        config=config,
    )

    print(f"Starting benchmark for {benchmark.system_name}...")
    benchmark.run()
    print(f"Benchmark finished. Results saved to {config.results_file}")

    volume.commit()


@app.function(
    image=neo4j_image,
    volumes={f"/{VOLUME_NAME}": volume},
    timeout=3600,
)
async def launch_neo4j_and_run_benchmark(config_params: dict, dir_suffix: str):
    """Launches Neo4j and then triggers the Graphiti benchmark."""
    print("Starting Neo4j server process...")
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

    neo4j_process = subprocess.Popen(
        "su-exec neo4j:neo4j neo4j console",
        shell=True,
    )

    print("Waiting for Neo4j server to become available on port 7474...")
    while True:
        try:
            with socket.create_connection(("localhost", 7474), timeout=1):
                print("✅ Neo4j server is ready.")
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

        bolt_host, bolt_port = bolt_tunnel.tcp_socket
        bolt_addr = f"bolt://{bolt_host}:{bolt_port}"
        user, password = neo4j_env_dict["NEO4J_AUTH"].split("/")

        print(f"🔌 Bolt address for this run: {bolt_addr}")

        # Update config with live credentials
        config_params["db_url"] = bolt_addr
        config_params["db_user"] = user
        config_params["db_password"] = password

        # Run the benchmark in a separate container and wait for it
        print("Running benchmark worker container and waiting for completion...")
        await run_graphiti_benchmark.remote.aio(config_params, dir_suffix)

    print("Benchmark worker finished. Shutting down Neo4j container.")
    neo4j_process.terminate()


@app.local_entrypoint()
async def main(
    runs: int = 2,
    corpus_limit: int = 2,
    qa_limit: int = 2,
    print_results: bool = True,
    model_name: str = "gpt-4o",
):
    """Trigger Graphiti QA benchmark runs on Modal."""
    print(f"🚀 Launchi  ng {runs} Graphiti QA benchmark run(s) on Modal...")

    # Generate unique timestamp for this benchmark session
    base_timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    config_params_list = []

    for run_num in range(runs):
        # Construct parameters dictionary directly
        config_params = {
            "corpus_limit": corpus_limit,
            "qa_limit": qa_limit,
            "print_results": print_results,
            "model_name": model_name,
        }

        # Create unique filename for this run
        unique_filename = f"run_{run_num + 1:03d}.json"
        config_params["results_file"] = (
            f"/{VOLUME_NAME}/{BENCHMARK_NAME}_{base_timestamp}/answers/{unique_filename}"
        )

        config_params_list.append(config_params)

    # Run benchmarks concurrently
    import asyncio

    tasks = []
    for params in config_params_list:
        print(f"Executing run, saving results to {params['results_file']}")
        task = launch_neo4j_and_run_benchmark.remote.aio(params, base_timestamp)
        tasks.append(task)

    await asyncio.gather(*tasks)

    print(f"✅ {runs} benchmark task(s) completed successfully.")

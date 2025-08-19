import asyncio
import datetime
import os
from dataclasses import asdict
from pathlib import Path

import modal

from qa.qa_benchmark_lightrag import LightRAGConfig, QABenchmarkLightRAG
from modal_apps.modal_image import image

APP_NAME = "qa-benchmark-lightrag"
VOLUME_NAME = "qa-benchmarks"
BENCHMARK_NAME = "lightrag"
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

app = modal.App(APP_NAME, image=image, secrets=[modal.Secret.from_dotenv()])


@app.function(
    volumes={f"/{VOLUME_NAME}": volume},
    timeout=86000,
    cpu=4,
    memory=16384,
)
def run_lightrag_benchmark(config_params: dict, dir_suffix: str):
    """Run the LightRAG QA benchmark on Modal."""
    print("Received benchmark request for LightRAG.")

    # Create benchmark folder structure
    answers_folder = _create_benchmark_folder(VOLUME_NAME, BENCHMARK_NAME, dir_suffix)
    print(f"Created benchmark folder: {answers_folder}")

    config = LightRAGConfig(**config_params)

    benchmark = QABenchmarkLightRAG.from_jsons(
        corpus_file=f"/root/{CORPUS_FILE.name}",
        qa_pairs_file=f"/root/{QA_PAIRS_FILE.name}",
        config=config,
    )

    print(f"Starting benchmark for {benchmark.system_name}...")
    benchmark.run()
    print(f"Benchmark finished. Results saved to {config.results_file}")

    volume.commit()


@app.local_entrypoint()
async def main(
    runs: int = 45,
    corpus_limit: int = None,
    qa_limit: int = None,
    query_mode: str = "hybrid",
    print_results: bool = True,
):
    """Trigger LightRAG QA benchmark runs on Modal."""
    print(f"🚀 Launching {runs} LightRAG QA benchmark run(s) on Modal...")

    # Generate unique timestamp for this benchmark session
    base_timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    config_params_list = []

    for run_num in range(runs):
        config = LightRAGConfig(
            corpus_limit=corpus_limit,
            qa_limit=qa_limit,
            query_mode=query_mode,
            print_results=print_results,
        )
        config_params = asdict(config)

        # Create unique filename for this run
        unique_filename = f"run_{run_num + 1:03d}.json"
        config_params["results_file"] = (
            f"/{VOLUME_NAME}/{BENCHMARK_NAME}_{base_timestamp}/answers/{unique_filename}"
        )

        config_params_list.append(config_params)

    # Fire-and-forget approach with 30s wait
    for params in config_params_list:
        run_lightrag_benchmark.spawn(params, base_timestamp)

    print(f"✅ {runs} benchmark task(s) submitted successfully.")

import asyncio
import datetime
import os
from dataclasses import asdict
from pathlib import Path

import modal

from qa.qa_benchmark_cognee import CogneeConfig, QABenchmarkCognee
from modal_apps.modal_image import image

APP_NAME = "qa-benchmark-cognee"
VOLUME_NAME = "qa-benchmarks"
BENCHMARK_NAME = "cognee"
QA_PAIRS_FILE = Path("hotpot_qa_24_qa_pairs.json")
INSTANCE_FILTER_FILE = Path("hotpot_qa_24_instance_filter.json")


def _create_benchmark_folder(
    volume_name: str, benchmark_name: str, timestamp: str, qa_engine: str
) -> str:
    """Create benchmark folder structure and return the answers folder path."""
    benchmark_folder = f"/{volume_name}/{benchmark_name}_{qa_engine}_{timestamp}"
    answers_folder = f"{benchmark_folder}/answers"

    # Create directories if they don't exist
    os.makedirs(answers_folder, exist_ok=True)

    return answers_folder


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image, secrets=[modal.Secret.from_dotenv()])


@app.function(
    volumes={f"/{VOLUME_NAME}": volume},
    timeout=3600,
    cpu=4,
    memory=16384,
)
def run_cognee_benchmark(config_params: dict, dir_suffix: str):
    """Run the Cognee QA benchmark on Modal."""
    print("Received benchmark request for Cognee.")

    # Create benchmark folder structure
    qa_engine = config_params.get("qa_engine", "cognee_graph_completion")
    answers_folder = _create_benchmark_folder(VOLUME_NAME, BENCHMARK_NAME, dir_suffix, qa_engine)
    print(f"Created benchmark folder: {answers_folder}")

    config = CogneeConfig(**config_params)

    benchmark = QABenchmarkCognee.from_jsons(
        qa_pairs_file=f"/root/{QA_PAIRS_FILE.name}",
        instance_filter_file=f"/root/{INSTANCE_FILTER_FILE.name}",
        config=config,
    )

    print(f"Starting benchmark for {benchmark.system_name}...")
    benchmark.run()
    print(f"Benchmark finished. Results saved to {config.results_file}")

    volume.commit()


# qa_engine: str = 'cognee_graph_completion', 'cognee_graph_completion_cot', 'cognee_graph_completion_context_extension'
@app.local_entrypoint()
async def main(
    runs: int = 45,
    corpus_limit: int = None,
    qa_limit: int = None,
    qa_engine: str = "cognee_graph_completion",  # 'cognee_graph_completion_cot', 'cognee_graph_completion_context_extension'
    top_k: int = 15,
    system_prompt_path: str = "answer_simple_question_benchmark2.txt",
    clean_start: bool = True,
    print_results: bool = True,
):
    """Trigger Cognee QA benchmark runs on Modal."""
    print(f"🚀 Launching {runs} Cognee QA benchmark run(s) on Modal with these parameters:")
    print(f"  - runs: {runs}")
    print(f"  - corpus_limit: {corpus_limit}")
    print(f"  - qa_limit: {qa_limit}")
    print(f"  - qa_engine: {qa_engine}")
    print(f"  - top_k: {top_k}")
    print(f"  - system_prompt_path: {system_prompt_path}")
    print(f"  - clean_start: {clean_start}")
    print(f"  - print_results: {print_results}")

    # Generate unique timestamp for this benchmark session
    base_timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    config_params_list = []

    for run_num in range(runs):
        config = CogneeConfig(
            corpus_limit=corpus_limit,
            qa_limit=qa_limit,
            qa_engine=qa_engine,
            top_k=top_k,
            system_prompt_path=system_prompt_path,
            clean_start=clean_start,
            print_results=print_results,
        )
        config_params = asdict(config)

        # Create unique filename for this run
        unique_filename = f"run_{run_num + 1:03d}.json"
        config_params["results_file"] = (
            f"/{VOLUME_NAME}/{BENCHMARK_NAME}_{qa_engine}_{base_timestamp}/answers/{unique_filename}"
        )

        config_params_list.append(config_params)

    # Fire-and-forget approach with 30s wait
    for params in config_params_list:
        run_cognee_benchmark.spawn(params, base_timestamp)

    print(f"✅ {runs} benchmark task(s) submitted successfully.")

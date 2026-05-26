"""
Batch Add + Cognify Performance Test

Starts the Cognee server, creates an API key, adds 200 files to a dataset,
calls cognify, and logs timing for every operation.

Usage:
    python -m cognee.tests.performance.batch_add_cognify_test
"""

import io
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import requests

TOPICS = [
    "quantum computing",
    "machine learning",
    "climate change",
    "renewable energy",
    "space exploration",
    "genetic engineering",
    "blockchain technology",
    "artificial intelligence",
    "ocean conservation",
    "urban planning",
    "medieval history",
    "philosophy of mind",
    "distributed systems",
    "neuroscience",
    "economic theory",
]

SUBTOPICS = [
    "data analysis",
    "pattern recognition",
    "resource allocation",
    "risk assessment",
    "optimization algorithms",
    "predictive modeling",
    "system integration",
    "scalability",
    "error correction",
    "signal processing",
    "network topology",
    "feedback loops",
    "energy efficiency",
    "material science",
    "behavioral adaptation",
    "information theory",
]

SENTENCE_TEMPLATES = [
    "The field of {topic} has seen remarkable advances in recent years, particularly in the area of {subtopic}.",
    "Researchers studying {topic} have discovered that {subtopic} plays a crucial role in understanding the broader implications.",
    "A comprehensive review of {topic} literature reveals that {subtopic} remains one of the most debated aspects.",
    "Recent experiments in {topic} demonstrate a strong correlation between {subtopic} and observed outcomes.",
    "The intersection of {topic} and {subtopic} opens new possibilities for practical applications.",
    "Experts in {topic} argue that {subtopic} will be the defining challenge of the next decade.",
    "Historical analysis shows that {topic} has always been influenced by developments in {subtopic}.",
    "New computational models for {topic} suggest that {subtopic} can be optimized through iterative approaches.",
    "The economic impact of {topic} is closely tied to advancements in {subtopic}, according to recent studies.",
    "Collaborative efforts in {topic} have led to breakthroughs in {subtopic} that were previously thought impossible.",
    "Understanding {topic} requires a deep appreciation of how {subtopic} interacts with existing frameworks.",
    "Policy makers are increasingly turning to {topic} research to inform decisions about {subtopic}.",
]


def generate_paragraph(topic: str, num_sentences: int = 5) -> str:
    sentences = []
    for _ in range(num_sentences):
        template = random.choice(SENTENCE_TEMPLATES)
        subtopic = random.choice(SUBTOPICS)
        sentences.append(template.format(topic=topic, subtopic=subtopic))
    return " ".join(sentences)


def generate_document(num_paragraphs: int = 3) -> tuple:
    topic = random.choice(TOPICS)
    paragraphs = [
        generate_paragraph(topic, num_sentences=random.randint(50, 100))
        for _ in range(num_paragraphs)
    ]
    paragraphs.append(str(uuid.uuid4()))
    return "\n\n".join(paragraphs), topic


NUM_FILES = 200
DATASET_NAME = f"batch_test_{uuid.uuid4().hex[:8]}"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def wait_for_server(url: str, timeout: float = 240.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    raise SystemExit(f"Server at {url} did not become ready in {timeout}s")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


def add_files_batch(base_url: str, api_key: str, count: int) -> float:
    log(f"  Generating {count} documents...")
    files = []
    for i in range(1, count + 1):
        text, _ = generate_document(num_paragraphs=random.randint(2, 5))
        files.append(
            ("data", (f"document_{i}.txt", io.BytesIO(text.encode("utf-8")), "text/plain"))
        )
    log(f"  Uploading {count} files in a single request...")

    start = time.time()
    resp = requests.post(
        f"{base_url}/api/v1/add",
        data={"datasetName": DATASET_NAME},
        files=files,
        headers={"X-Api-Key": api_key},
        timeout=1800,
    )
    elapsed = time.time() - start

    if resp.status_code != 200:
        log(f"  ERROR: {resp.status_code} - {resp.text[:300]}")
    return elapsed


def cognify(base_url: str, api_key: str) -> float:
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}
    payload = {"datasets": [DATASET_NAME], "data_per_batch": NUM_FILES, "runInBackground": False}

    start = time.time()
    resp = requests.post(
        f"{base_url}/api/v1/cognify",
        json=payload,
        headers=headers,
        timeout=36000,
    )
    elapsed = time.time() - start

    if resp.status_code != 200:
        log(f"  Cognify ERROR: {resp.status_code} - {resp.text[:300]}")
    return elapsed


def main() -> None:
    host = os.environ.get("HTTP_API_HOST", "localhost")
    port = os.environ.get("HTTP_API_PORT", "8000")
    base_url = f"http://{host}:{port}"

    perf_dir = str(Path(__file__).resolve().parent)
    key_path = tempfile.NamedTemporaryFile(suffix=".key", delete=False).name

    log("=== Bootstrapping: pruning data, creating user & API key ===")
    bootstrap_start = time.time()
    try:
        subprocess.run(
            [VENV_PYTHON, "-m", "utils.bootstrap_script", key_path],
            check=True,
            cwd=perf_dir,
        )
        api_key = Path(key_path).read_text().strip()
    finally:
        try:
            os.unlink(key_path)
        except FileNotFoundError:
            pass
    log(f"Bootstrap done in {time.time() - bootstrap_start:.1f}s")

    log(f"Starting Cognee server on {base_url}")
    server_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "cognee.api.client:app", "--host", host, "--port", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        wait_for_server(f"{base_url}/health")
        log("Server is ready")

        log(f"=== Adding {NUM_FILES} files to dataset '{DATASET_NAME}' (single request) ===")
        total_add_time = add_files_batch(base_url, api_key, NUM_FILES)
        log("=== Add phase complete ===")
        log(f"  Total:   {total_add_time:.1f}s")
        log(f"  Average: {total_add_time / NUM_FILES:.3f}s per file")

        log(f"=== Running cognify on dataset '{DATASET_NAME}' ===")
        cognify_time = cognify(base_url, api_key)
        log("=== Cognify complete ===")
        log(f"  Time: {cognify_time:.1f}s")

        log("=== Summary ===")
        log(f"  Files added:     {NUM_FILES}")
        log(
            f"  Add total time:  {total_add_time:.1f}s ({total_add_time / NUM_FILES:.3f}s avg per file)"
        )
        log(f"  Cognify time:    {cognify_time:.1f}s")
        log(f"  Total time:      {total_add_time + cognify_time:.1f}s")

    finally:
        log("Shutting down server")
        try:
            os.killpg(server_proc.pid, signal.SIGTERM)
            server_proc.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()

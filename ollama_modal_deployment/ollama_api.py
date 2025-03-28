import modal
import os
import subprocess
import time
from fastapi import FastAPI, HTTPException
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
import ollama
from fastapi.middleware.cors import CORSMiddleware

import httpx
from fastapi import Request, Response

MODEL = os.environ.get("MODEL", "deepseek-r1:70b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "avr/sfr-embedding-mistral")


def pull() -> None:
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    wait_for_ollama()
    subprocess.run(["ollama", "pull", MODEL], stdout=subprocess.PIPE)
    subprocess.run(["ollama", "pull", EMBEDDING_MODEL], stdout=subprocess.PIPE)


def wait_for_ollama(timeout: int = 30, interval: int = 2) -> None:
    import httpx
    from loguru import logger

    start_time = time.time()
    while True:
        try:
            response = httpx.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                logger.info("Ollama service is ready")
                return
        except httpx.ConnectError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Ollama service failed to start")
            logger.info(f"Waiting for Ollama service... ({int(time.time() - start_time)}s)")
            time.sleep(interval)


image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands(  # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .copy_local_file("ollama.service", "/etc/systemd/system/ollama.service")
    .pip_install("ollama", "httpx", "loguru", "fastapi")
    # .env({"OLLAMA_MODELS": "/persistent/ollama-models"})
    # .run_function(check_blobs_directory)
    .run_function(pull)
)
app = modal.App(name="ollama", image=image)
api = FastAPI()


@api.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(full_path: str, request: Request):
    # Construct the local Ollama endpoint URL
    local_url = f"http://localhost:11434/{full_path}"
    print(f"Forwarding {request.method} request to: {local_url}")  # Logging the target URL
    # Forward the request
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        response = await client.request(
            method=request.method,
            url=local_url,
            headers=request.headers.raw,
            params=request.query_params,
            content=await request.body(),
        )
    print(f"Received response with status: {response.status_code}")  # Logging the response status
    return Response(
        content=response.content, status_code=response.status_code, headers=response.headers
    )


@app.cls(
    gpu="L40S:1",
    scaledown_window=5 * 60,
)
class Ollama:
    def __init__(self):
        self.serve()

    @modal.build()
    def build(self):
        subprocess.run(["systemctl", "daemon-reload"])
        subprocess.run(["systemctl", "enable", "ollama"])

    @modal.enter()
    def enter(self):
        subprocess.run(["systemctl", "start", "ollama"])
        wait_for_ollama()
        # subprocess.run(["ollama", "pull", MODEL])

    @modal.asgi_app()
    def serve(self):
        return api

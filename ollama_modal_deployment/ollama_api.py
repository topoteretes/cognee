import modal
import os
import subprocess
import time
from fastapi import FastAPI, HTTPException
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
import ollama

MODEL = os.environ.get("MODEL", "llama3.3:70b")


def pull() -> None:
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    wait_for_ollama()
    subprocess.run(["ollama", "pull", MODEL], stdout=subprocess.PIPE)


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


class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g. 'user', 'assistant')")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=MODEL, description="The model to use for completion")
    messages: List[ChatMessage] = Field(
        ..., description="The messages to generate a completion for"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")
    format: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "A JSON dictionary specifying any kind of structured output expected. "
            "For example, it can define a JSON Schema to validate the response."
        ),
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional options for the model (e.g., temperature, etc.)."
    )


@api.post("/v1/api/chat")
async def v1_chat_completions(request: ChatCompletionRequest) -> Any:
    try:
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail="Messages array is required and cannot be empty",
            )
        response = ollama.chat(
            model=request.model,
            messages=[msg for msg in request.messages],
            stream=request.stream,
            format=request.format,
            options=request.options,
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")


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

"""Docker container lifecycle for per-dataset Neo4j Community instances.

Neo4j Community edition supports exactly ONE database per running server —
``CREATE DATABASE`` is Enterprise-only. Per-dataset graph isolation on the free
edition therefore means one server per dataset, realized here as one Docker
container per dataset (structurally the same model as
``Neo4jAuraDevDatasetDatabaseHandler``'s one-Aura-instance-per-dataset, with
Docker instead of the Aura REST API).

Lifecycle contract (driven by ``Neo4jCommunityDatasetDatabaseHandler`` and the
graph-engine cache):

- ``create_container`` — provision a named container with a named data volume
  and a fixed published host port; used once per dataset.
- ``ensure_running`` — called from ``resolve_dataset_connection_info`` before
  every operation on the dataset; starts the container if it is stopped
  (data persists on the named volume) and waits for bolt readiness.
- ``stop_container_for_url`` — called from ``Neo4jCommunityAdapter.close``,
  which the ``closing_lru_cache`` invokes on engine eviction; idle datasets
  release their containers automatically.
- ``remove_container`` — called from the handler's ``delete_dataset``; removes
  the container and its volume.
- The number of concurrently RUNNING containers is bounded by
  ``NEO4J_COMMUNITY_MAX_CONTAINERS`` (default: ``DATABASE_MAX_LRU_CACHE_SIZE``);
  before a container starts, least-recently-used running ones are stopped.

This module deliberately has no neo4j / heavy imports so it is safe to import
from the eagerly-loaded dataset-database-handler registry.
"""

import os
import shutil
import socket
import subprocess
import threading
import asyncio
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from cognee.shared.logging_utils import get_logger
from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE

logger = get_logger("Neo4jCommunityContainers")

# Label identifying containers managed by this module. Every docker listing is
# filtered on it so cognee never touches containers it did not create.
COGNEE_CONTAINER_LABEL = "ai.cognee.neo4j_community"
COGNEE_DATASET_LABEL = "ai.cognee.dataset_id"

CONTAINER_NAME_PREFIX = "cognee-neo4j-"
VOLUME_NAME_PREFIX = "cognee-neo4j-data-"

# Generous timeouts: the first `docker run` may pull the image.
DOCKER_RUN_TIMEOUT_SECONDS = 600
DOCKER_COMMAND_TIMEOUT_SECONDS = 120

# The bolt handshake magic preamble followed by four proposed protocol
# versions (5.4, 4.4, 3, 2). Any 4-byte reply proves a live bolt server is
# answering on the port. A plain TCP connect is NOT a readiness signal:
# docker-proxy accepts connections on the published port as soon as the
# container starts, long before Neo4j listens inside it.
_BOLT_HANDSHAKE = (
    b"\x60\x60\xb0\x17\x00\x00\x04\x05\x00\x00\x04\x04\x00\x00\x00\x03\x00\x00\x00\x02"
)


def get_neo4j_community_image() -> str:
    return os.environ.get("NEO4J_COMMUNITY_IMAGE", "neo4j:5-community")


def get_max_running_containers() -> int:
    """Ceiling on concurrently running per-dataset Neo4j containers.

    Defaults to ``DATABASE_MAX_LRU_CACHE_SIZE`` so the container pool tracks
    the graph-engine LRU cache: an engine past the cache capacity is evicted
    and closed, which stops its container anyway.
    """
    raw = os.environ.get("NEO4J_COMMUNITY_MAX_CONTAINERS", "")
    if raw.strip():
        return max(1, int(raw))
    return max(1, int(DATABASE_MAX_LRU_CACHE_SIZE))


def get_startup_timeout_seconds() -> float:
    return float(os.environ.get("NEO4J_COMMUNITY_STARTUP_TIMEOUT", "120"))


def check_docker_available() -> Tuple[bool, str]:
    """Check that the Docker CLI exists and the daemon responds.

    Same preflight contract as ``cognee.api.v1.ui.ui._check_docker_available``
    (not imported from there: infrastructure must not depend on the API layer).
    """
    if shutil.which("docker") is None:
        return False, (
            "Docker CLI not found on PATH. The neo4j_community dataset database handler "
            "runs one Neo4j Community container per dataset and requires Docker. "
            "Install Docker Desktop (https://www.docker.com/products/docker-desktop/) "
            "or Colima and ensure the `docker` binary is on your PATH."
        )

    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
    except subprocess.TimeoutExpired:
        return False, (
            "Docker daemon did not respond within 15 seconds. "
            "The daemon may be starting up; wait a moment and try again."
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"Docker CLI could not be executed ({type(error).__name__}: {error})."

    if result.returncode == 0:
        return True, "Docker daemon is running."

    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    return False, (
        "Docker CLI is installed but the daemon is not responding "
        f"(docker info stderr: {stderr}). Start Docker Desktop, run `colima start`, "
        "or `sudo systemctl start docker` depending on your setup."
    )


def container_name_for_dataset(dataset_id) -> str:
    from uuid import UUID

    return f"{CONTAINER_NAME_PREFIX}{UUID(str(dataset_id)).hex}"


def volume_name_for_dataset(dataset_id) -> str:
    from uuid import UUID

    return f"{VOLUME_NAME_PREFIX}{UUID(str(dataset_id)).hex}"


def allocate_free_port() -> int:
    """Ask the OS for a currently-free TCP port on localhost.

    Subject to the usual time-of-check race; callers retry container creation
    with a fresh port when docker reports the port as already allocated.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def bolt_url_for_port(host_port: int) -> str:
    return f"bolt://localhost:{host_port}"


def _run_docker_sync(args: List[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, timeout=timeout)


async def _docker(args: List[str], timeout: float = DOCKER_COMMAND_TIMEOUT_SECONDS) -> str:
    """Run a docker CLI command off-loop and return stdout; raise on failure."""
    result = await asyncio.to_thread(_run_docker_sync, args, timeout)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"`docker {' '.join(args)}` failed: {stderr}")
    return result.stdout.decode("utf-8", errors="replace").strip()


def _bolt_ready(host_port: int, timeout: float = 2.0) -> bool:
    """True when a bolt server answers the protocol handshake on the port."""
    try:
        with socket.create_connection(("localhost", host_port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(_BOLT_HANDSHAKE)
            response = sock.recv(4)
            return len(response) == 4
    except OSError:
        return False


class Neo4jCommunityContainerManager:
    """Owns the run-state of cognee-managed Neo4j Community containers.

    In-process bookkeeping (the recency order and the bolt-url -> container
    map) is advisory: every decision that matters re-checks Docker itself, so
    containers surviving a Cognee restart are found, restarted, and counted
    against the ceiling like any other.
    """

    def __init__(self):
        # container name -> bolt url, most recently used last.
        self._recency: "OrderedDict[str, str]" = OrderedDict()
        self._url_to_container: Dict[str, str] = {}
        # Guards only the fast in-memory maps; never held across awaits.
        self._registry_lock = threading.Lock()

    def _touch(self, container_name: str, bolt_url: str) -> None:
        with self._registry_lock:
            self._recency.pop(container_name, None)
            self._recency[container_name] = bolt_url
            self._url_to_container[bolt_url] = container_name

    def _forget(self, container_name: str, bolt_url: Optional[str] = None) -> None:
        with self._registry_lock:
            registered_url = self._recency.pop(container_name, None)
            for url in {registered_url, bolt_url}:
                if url is not None and self._url_to_container.get(url) == container_name:
                    self._url_to_container.pop(url, None)

    def container_for_url(self, bolt_url: str) -> Optional[str]:
        with self._registry_lock:
            return self._url_to_container.get(bolt_url)

    async def _container_state(self, container_name: str) -> Optional[str]:
        """Return docker's state string ("running", "exited", ...) or None
        when no container with that name exists."""
        result = await asyncio.to_thread(
            _run_docker_sync,
            ["inspect", "--format", "{{.State.Status}}", container_name],
            DOCKER_COMMAND_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return None
        return result.stdout.decode("utf-8", errors="replace").strip()

    async def _running_managed_containers(self) -> List[str]:
        output = await _docker(
            [
                "ps",
                "--filter",
                f"label={COGNEE_CONTAINER_LABEL}=true",
                "--format",
                "{{.Names}}",
            ]
        )
        return [name for name in output.splitlines() if name]

    async def _enforce_ceiling(self, protected_container: str) -> None:
        """Stop least-recently-used running containers until there is room for
        one more. Containers unknown to this process (left over from a previous
        run) are treated as the oldest and stopped first."""
        running = await self._running_managed_containers()
        max_running = get_max_running_containers()
        slots_needed = len(running) - (max_running - 1)
        if slots_needed <= 0:
            return

        with self._registry_lock:
            recency_order = list(self._recency)
        known_order = {name: index for index, name in enumerate(recency_order)}
        victims = sorted(
            (name for name in running if name != protected_container),
            key=lambda name: known_order.get(name, -1),
        )[:slots_needed]

        for victim in victims:
            logger.info(
                "Neo4j Community container ceiling (%s) reached; stopping least-recently-used "
                "container %s",
                max_running,
                victim,
            )
            await self.stop_container(victim)

    async def stop_container(self, container_name: str) -> None:
        """Stop a managed container and evict any cached graph engine bound to
        it, so no live engine keeps talking to a stopped server."""
        with self._registry_lock:
            bolt_url = self._recency.get(container_name)
        self._forget(container_name)

        if bolt_url is not None:
            # Lazy import: get_graph_engine lazily imports the community adapter,
            # which imports this module.
            from cognee.infrastructure.databases.graph.get_graph_engine import (
                evict_graph_engines_for_url,
            )

            evict_graph_engines_for_url(bolt_url)

        try:
            await _docker(["stop", container_name])
        except RuntimeError as error:
            # Stopping an already-stopped or externally-removed container is
            # not an error worth failing an operation over.
            logger.debug("Failed to stop container %s: %s", container_name, error)

    async def stop_container_for_url(self, bolt_url: str) -> None:
        """Stop the container serving ``bolt_url``. Invoked by
        ``Neo4jCommunityAdapter.close`` when the graph-engine cache evicts the
        dataset's engine. Unknown urls are a no-op (e.g. the container was
        already stopped or removed)."""
        container_name = self.container_for_url(bolt_url)
        if container_name is None:
            return
        self._forget(container_name, bolt_url)
        try:
            await _docker(["stop", container_name])
            logger.info("Stopped idle Neo4j Community container %s", container_name)
        except RuntimeError as error:
            logger.debug("Failed to stop container %s: %s", container_name, error)

    async def create_container(
        self,
        dataset_id,
        container_name: str,
        volume_name: str,
        host_port: int,
        password: str,
    ) -> None:
        """Create and start the dataset's container with a named data volume.

        The host port is published on 127.0.0.1 only and is fixed for the
        container's lifetime (``docker start`` re-publishes the same port), so
        the bolt url stored in the DatasetDatabase row stays valid.
        """
        await self._enforce_ceiling(protected_container=container_name)

        await _docker(
            [
                "run",
                "--detach",
                "--name",
                container_name,
                "--label",
                f"{COGNEE_CONTAINER_LABEL}=true",
                "--label",
                f"{COGNEE_DATASET_LABEL}={dataset_id}",
                "--publish",
                f"127.0.0.1:{host_port}:7687",
                "--volume",
                f"{volume_name}:/data",
                "--env",
                f"NEO4J_AUTH=neo4j/{password}",
                get_neo4j_community_image(),
            ],
            timeout=DOCKER_RUN_TIMEOUT_SECONDS,
        )
        await self.wait_for_bolt(container_name, host_port)
        self._touch(container_name, bolt_url_for_port(host_port))

    async def ensure_running(self, container_name: str, host_port: int) -> None:
        """Start the dataset's container if it is not running and wait until
        bolt answers. Called before every operation on the dataset."""
        bolt_url = bolt_url_for_port(host_port)

        # Fast path: we believe it is running and a bolt server answers.
        if self.container_for_url(bolt_url) == container_name and _bolt_ready(host_port):
            self._touch(container_name, bolt_url)
            return

        state = await self._container_state(container_name)
        if state is None:
            raise RuntimeError(
                f"Neo4j Community container '{container_name}' does not exist (it may have "
                "been removed outside of cognee). Delete and re-create the dataset to "
                "provision a new container."
            )

        if state != "running":
            await self._enforce_ceiling(protected_container=container_name)
            try:
                await _docker(["start", container_name])
            except RuntimeError as error:
                raise RuntimeError(
                    f"Failed to start Neo4j Community container '{container_name}' on port "
                    f"{host_port}. If the port is now taken by another process, free it and "
                    f"retry. Underlying error: {error}"
                ) from error

        await self.wait_for_bolt(container_name, host_port)
        self._touch(container_name, bolt_url)

    async def remove_container(
        self, container_name: str, volume_name: str, bolt_url: Optional[str] = None
    ) -> None:
        """Remove the dataset's container and its data volume (teardown)."""
        self._forget(container_name, bolt_url)

        try:
            await _docker(["rm", "--force", container_name])
        except RuntimeError as error:
            logger.debug("Failed to remove container %s: %s", container_name, error)
        try:
            await _docker(["volume", "rm", "--force", volume_name])
        except RuntimeError as error:
            logger.debug("Failed to remove volume %s: %s", volume_name, error)

    async def wait_for_bolt(self, container_name: str, host_port: int) -> None:
        timeout = get_startup_timeout_seconds()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if await asyncio.to_thread(_bolt_ready, host_port):
                return
            await asyncio.sleep(1)
        raise TimeoutError(
            f"Neo4j Community container '{container_name}' did not accept bolt connections on "
            f"port {host_port} within {timeout:.0f} seconds. Check "
            f"`docker logs {container_name}` for startup errors."
        )


_container_manager: Optional[Neo4jCommunityContainerManager] = None
_container_manager_lock = threading.Lock()


def get_container_manager() -> Neo4jCommunityContainerManager:
    """Process-wide singleton manager."""
    global _container_manager
    if _container_manager is None:
        with _container_manager_lock:
            if _container_manager is None:
                _container_manager = Neo4jCommunityContainerManager()
    return _container_manager

# Docker & Colima Setup for Cognee UI / MCP

The `cognee-cli -ui` command starts an MCP server inside a Docker container.
This requires a running Docker-compatible daemon. Both **Docker Desktop** and
**Colima** (an open-source, commercially-free alternative) are supported.

## Option A: Docker Desktop

Install from <https://www.docker.com/products/docker-desktop/> and start the
application. No extra configuration is needed.

## Option B: Colima (macOS / Linux)

[Colima](https://github.com/abiosoft/colima) provides a lightweight container
runtime without a Docker Desktop licence.

### Install

```bash
# macOS (Homebrew)
brew install colima docker

# Linux (Homebrew)
brew install colima docker
```

### Start Colima

```bash
# Basic start
colima start

# Recommended: enable host.docker.internal hostname resolution
colima start --network-address
```

> **Important:** Without `--network-address`, the hostname
> `host.docker.internal` may not resolve inside containers. The cognee MCP
> entrypoint includes automatic fallback logic (tries `host.docker.internal`,
> then `host.lima.internal`, then the container's default gateway IP), but
> starting Colima with `--network-address` avoids these workarounds entirely.

### Verify

```bash
docker info   # Should print server information without errors
docker run --rm hello-world
```

## Troubleshooting

### "Docker daemon is not responding"

| Runtime        | Fix                                                                    |
|----------------|------------------------------------------------------------------------|
| Docker Desktop | Open the Docker Desktop application and wait for the engine to start.  |
| Colima         | Run `colima start` (or `colima start --network-address`).              |
| Linux systemd  | Run `sudo systemctl start docker`.                                     |

### Container cannot reach host API (`localhost` / `127.0.0.1`)

Inside a container, `localhost` refers to the container itself, not the host
machine. The MCP entrypoint automatically rewrites `localhost` / `127.0.0.1`
to a reachable host address using the following fallback order:

1. `host.docker.internal` (Docker Desktop on macOS / Windows / Linux)
2. `host.lima.internal` (Colima / Lima)
3. Default gateway IP (plain Linux Docker, typically `172.17.0.1`)

If none of these work:

```bash
# Use the Docker bridge gateway directly
docker run -e API_URL=http://172.17.0.1:8000 ...

# Or use host networking (Linux only)
docker run --network host ...
```

### Colima: `host.docker.internal` does not resolve

Start Colima with `--network-address`:

```bash
colima stop
colima start --network-address
```

This adds the `host.docker.internal` DNS entry inside the VM.

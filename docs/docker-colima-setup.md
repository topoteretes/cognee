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

# Recommended: give the VM a host-reachable network address
colima start --network-address
```

> **Important:** `--network-address` does not itself add a `host.docker.internal`
> DNS entry — it provisions a shared-network IP that is reachable from both the
> Colima VM and the host. Colima maps `host.docker.internal` →
> `host.lima.internal` by default (recent versions), so combined with a
> reachable address `host.docker.internal` generally resolves; directly
> resolving it to the `--network-address` IP is still an open request
> ([abiosoft/colima#560](https://github.com/abiosoft/colima/issues/560)). The
> cognee MCP entrypoint also includes automatic fallback logic (tries
> `host.docker.internal`, then `host.lima.internal`, then the container's
> default gateway IP), so the host-API flow works even without this flag.

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

First, give the VM a host-reachable address (most setups need only this):

```bash
colima stop
colima start --network-address
```

This provisions a VM IP reachable from the host; `host.docker.internal` then
typically resolves via Colima's default `host.docker.internal` →
`host.lima.internal` mapping. Note that `--network-address` does **not** itself
write a DNS entry.

If the name still does not resolve (e.g. a missing or custom mapping), add it
explicitly via `network.dnsHosts` in `~/.colima/default/colima.yaml`:

```yaml
network:
  dnsHosts:
    host.docker.internal: host.lima.internal
```

then `colima restart`.

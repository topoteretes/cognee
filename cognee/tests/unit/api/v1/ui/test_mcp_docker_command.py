from cognee.api.v1.ui.ui import build_mcp_docker_command


def test_build_mcp_docker_command_uses_host_gateway_and_host_docker_internal():
    cmd = build_mcp_docker_command(
        container_name="cognee-mcp-test",
        mcp_port=8001,
        image="cognee/cognee-mcp:main",
        backend_port=8000,
    )

    assert cmd[:2] == ["docker", "run"]
    assert "--add-host" in cmd
    assert "host.docker.internal:host-gateway" in cmd
    assert "API_URL=http://host.docker.internal:8000" in cmd
    assert cmd[-1] == "cognee/cognee-mcp:main"


def test_build_mcp_docker_command_direct_mode_uses_env_file():
    cmd = build_mcp_docker_command(
        container_name="cognee-mcp-test",
        mcp_port=8001,
        image="cognee/cognee-mcp:main",
        env_file="/tmp/.env",
    )

    assert "--env-file" in cmd
    assert "/tmp/.env" in cmd
    assert "API_URL=" not in " ".join(cmd)

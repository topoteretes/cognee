import os
import tomlkit

def test_pylance_dependency():
    """Verify that pylance is not listed under project.dependencies but is in optional-dependencies.dev."""
    pyproject_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../pyproject.toml")
    )

    assert os.path.exists(pyproject_path), f"pyproject.toml not found at {pyproject_path}"

    with open(pyproject_path, "r", encoding="utf-8") as f:
        config = tomlkit.parse(f.read())

    project = config.get("project", {})
    dependencies = project.get("dependencies", [])

    pylance_in_deps = any("pylance" in dep for dep in dependencies)
    assert not pylance_in_deps, "pylance should not be in the runtime dependencies"

    optional_deps = project.get("optional-dependencies", {})
    dev_deps = optional_deps.get("dev", [])

    pylance_in_dev = any("pylance" in dep for dep in dev_deps)
    assert pylance_in_dev, "pylance should be in optional-dependencies.dev"

from pathlib import Path


def test_pylance_is_dev_optional_dependency_only():
    pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")

    assert "pylance>=0.22.0,<=0.36.0" in pyproject_text

    current_section = None
    in_dependencies = False
    in_dev = False
    runtime_dependencies = []
    dev_dependencies = []

    for line in pyproject_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped
            in_dependencies = False
            in_dev = False
            continue

        if current_section == "[project]":
            if stripped.startswith("dependencies") and stripped.endswith("["):
                in_dependencies = True
                continue
            if in_dependencies:
                if stripped.startswith("]"):
                    in_dependencies = False
                else:
                    runtime_dependencies.append(stripped)

        elif current_section == "[project.optional-dependencies]":
            if stripped.startswith("dev") and stripped.endswith("["):
                in_dev = True
                continue
            if in_dev:
                if stripped.startswith("]"):
                    in_dev = False
                else:
                    dev_dependencies.append(stripped)

    assert any("pylance>=0.22.0,<=0.36.0" in dep for dep in dev_dependencies), (
        "Expected pylance to be declared in [project.optional-dependencies].dev"
    )
    assert not any("pylance>=0.22.0,<=0.36.0" in dep for dep in runtime_dependencies), (
        "Expected pylance to be removed from the [project].dependencies runtime list"
    )

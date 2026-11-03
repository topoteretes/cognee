"""The crates.io source reads one version per crate through mocked HTTP, never the live API."""

import httpx
import pytest
from dlt.extract.exceptions import ResourceExtractionError

from cognee.tasks.ingestion.connectors import crates_source
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

CRATES = {
    "tokio": {
        "crate": {
            "max_stable_version": "1.38.0",
            "max_version": "1.39.0-rc1",
            "description": "An event-driven runtime.",
            "documentation": "https://docs.rs/tokio",
            "repository": "https://github.com/tokio-rs/tokio",
            "keywords": ["async", "network"],
        },
        "versions": [
            {"num": "1.39.0-rc1", "license": "MIT", "features": {}},
            {
                "num": "1.38.0",
                "license": "MIT",
                "features": {"default": [], "rt": [], "full": ["rt"]},
            },
            {"num": "1.0.0", "license": "MIT", "features": {}},
        ],
    },
    "serde": {
        "crate": {"max_stable_version": "1.0.200", "max_version": "1.0.200", "keywords": []},
        "versions": [{"num": "1.0.200", "license": "MIT OR Apache-2.0", "features": {}}],
    },
}


def _handler(seen):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        parts = request.url.path.strip("/").split("/")
        name = parts[3]
        if name not in CRATES:
            return httpx.Response(404)
        if len(parts) == 4:
            return httpx.Response(200, json=CRATES[name])
        if parts[5] == "dependencies":
            return httpx.Response(
                200,
                json={
                    "dependencies": [
                        {
                            "crate_id": "bytes",
                            "req": "^1",
                            "kind": "normal",
                            "optional": True,
                            "features": ["serde"],
                        }
                    ]
                },
            )
        return httpx.Response(200, text=f"<h1>{name}</h1><p>Readme of <b>{name}</b></p>")

    return handle


def _rows(seen, **kwargs):
    client = httpx.Client(
        transport=httpx.MockTransport(_handler(seen)), headers={"User-Agent": "test-agent"}
    )
    return list(crates_source(client=client, request_interval=0, **kwargs))


def test_reads_latest_stable_version_with_features_dependencies_and_readme():
    seen = []
    (row,) = _rows(seen, crate_names=["tokio"])
    assert row["title"] == "tokio 1.38.0"
    assert "- full: rt" in row["content"]
    assert "License: MIT" in row["content"]
    assert "Keywords: async, network" in row["content"]
    assert "- bytes ^1 (normal; optional; features: serde)" in row["content"]
    assert "Readme of tokio" in row["content"] and "<b>" not in row["content"]
    assert all("1.39.0-rc1" not in str(r.url) for r in seen)


def test_requested_version_and_stable_content_hash_id():
    (first,) = _rows([], crate_names=["tokio@1.0.0"])
    (second,) = _rows([], crate_names=["tokio@1.0.0"])
    assert first["title"] == "tokio 1.0.0"
    assert first["id"] == second["id"]


def test_cargo_toml_resolves_registry_dependencies(tmp_path):
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text(
        '[dependencies]\ntokio = "1"\nlocal = { path = "../local" }\n'
        'ser = { package = "serde", version = "1" }\n'
    )
    titles = [row["title"] for row in _rows([], cargo_toml=manifest)]
    assert titles == ["serde 1.0.200", "tokio 1.38.0"]


def test_unknown_crate_and_unknown_version_raise():
    with pytest.raises(ResourceExtractionError, match="crates.io returned 404|no version"):
        _rows([], crate_names=["nope"])
    with pytest.raises(ResourceExtractionError, match="crates.io returned 404|no version"):
        _rows([], crate_names=["tokio@9.9.9"])


def test_every_request_carries_the_user_agent():
    seen = []
    _rows(seen, crate_names=["serde"])
    assert {r.headers["user-agent"] for r in seen} == {"test-agent"}


def test_declares_document_source_and_validates_input():
    resource = crates_source(crate_names=["serde"])
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "crates"
    with pytest.raises(ValueError):
        crates_source()
    with pytest.raises(ValueError):
        crates_source(crate_names="serde")
    with pytest.raises(ValueError):
        crates_source(crate_names=["serde"], user_agent=" ")

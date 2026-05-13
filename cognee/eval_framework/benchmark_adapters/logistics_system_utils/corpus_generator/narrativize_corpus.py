from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.utils.utils import (
    _entity_entries,
    _format_packages,
    _safe_filename,
    load_world,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.ontology import (
    pretty_print_world,
)
from cognee.infrastructure.llm import LLMGateway


BASE_PATH = Path(__file__).resolve().parent.parent
WORLD_ENTITY_FOLDERS = ("carrier", "post_office", "retailer", "user")
PACKAGE_FOLDER = "packages"
WORLD_FILENAME = "stored_world.json"
NARRATIVIZATION_PROMPT = BASE_PATH.joinpath("prompts", "narrativization.txt").read_text(
    encoding="utf-8"
)
PACKAGE_NARRATIVIZATION_PROMPT = BASE_PATH.joinpath(
    "prompts", "package_narrativization.txt"
).read_text(encoding="utf-8")


class NarrativeOutput(BaseModel):
    entities: dict[str, str]


class PackageNarrativeOutput(BaseModel):
    packages: dict[str, str]


def _save_packages(world_root_path: Path, package_response: PackageNarrativeOutput) -> None:
    package_folder = world_root_path / PACKAGE_FOLDER
    package_folder.mkdir(parents=True, exist_ok=True)
    for package_id, narrative in package_response.packages.items():
        package_folder.joinpath(f"{_safe_filename(package_id)}.txt").write_text(
            narrative,
            encoding="utf-8",
        )


def _save_world_model(
    world_root_path: Path,
    response: NarrativeOutput,
    entity_entries: dict[str, tuple[Path, str]],
) -> None:
    for key, value in response.entities.items():
        entity_entry = entity_entries.get(key) or entity_entries.get(key.lower())
        if entity_entry is None:
            raise ValueError(f"Received narrative for unknown entity id: {key}")
        folder_name, entity_id = entity_entry
        folder = world_root_path / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        folder.joinpath(f"{_safe_filename(entity_id)}.txt").write_text(value, encoding="utf-8")


async def _query_llm(text: str, query: str, response_model: type[BaseModel]) -> BaseModel:
    return await LLMGateway.acreate_structured_output(
        text_input=text,
        system_prompt=query,
        response_model=response_model,
    )


def _fallback_world_narratives(world: dict[str, object]) -> NarrativeOutput:
    entities: dict[str, str] = {}

    for retailer in world.get("retailers", []):
        entities[retailer.retailer_id] = (
            f"{retailer.name} is a retailer operating in {retailer.region.display_name}. "
            f"It handles {retailer.shipping_range.label} shipping, charges a handling fee of "
            f"{retailer.handling_fee}, takes about {retailer.processing_days} processing days, "
            f"and ships from post office {retailer.origin_post_office_id}."
        )

    for user in world.get("users", []):
        entities[user.user_id] = (
            f"{user.name} is a {user.tier.label} customer in {user.region.display_name}. "
            f"Weekend delivery eligibility is {user.weekend_delivery_eligible}, and the default "
            f"shipping range is {user.default_shipping_range.label}."
        )

    for post_office in world.get("post_offices", []):
        entities[post_office.post_office_id] = (
            f"{post_office.name} is a {post_office.office_type.label} in "
            f"{post_office.region.display_name}. It supports {post_office.shipping_range.label} "
            f"shipping, cold chain support is {post_office.supports_cold_chain}, and hazardous "
            f"materials support is {post_office.supports_hazardous_materials}."
        )

    for carrier in world.get("carriers", []):
        supported_modes = ", ".join(mode.label for mode in carrier.supported_modes)
        shipping_ranges = ", ".join(
            shipping_range.label for shipping_range in carrier.shipping_ranges
        )
        entities[carrier.carrier_id] = (
            f"{carrier.company_name} is a carrier in {carrier.region.display_name}. "
            f"It supports {supported_modes} transport across {shipping_ranges} shipping ranges. "
            f"Temperature controlled support is {carrier.temperature_controlled}, hazardous "
            f"certification is {carrier.hazardous_certified}, weekend operations is "
            f"{carrier.weekend_operations}, maximum weight is {carrier.max_weight_kg} kg, "
            f"reliability score is {carrier.reliability_score}, and base delay is "
            f"{carrier.base_delay_days} day(s)."
        )

    return NarrativeOutput(entities=entities)


def _fallback_package_narratives(world: dict[str, object]) -> PackageNarrativeOutput:
    packages: dict[str, str] = {}
    for package in world.get("packages", []):
        current_post_office = package.current_post_office_name or "none"
        route = ", ".join(package.route_post_office_names) or "none"
        packages[package.package_id] = (
            f"Package {package.package_id} contains {package.description}. "
            f"It is associated with retailer {package.retailer_name} and user {package.user_name}. "
            f"The package weighs {package.weight_kg} kg, uses {package.shipping_range.label} "
            f"shipping, belongs to the {package.category.label} category, has "
            f"{package.priority.label} priority, insured status is {package.insured}, current "
            f"state is {package.current_state.label}, last known location is "
            f"{package.last_known_location}, current post office is {current_post_office}, and "
            f"the route includes {route}."
        )

    return PackageNarrativeOutput(packages=packages)


def _entity_file_paths(world_root_path: Path, world: dict[str, object]) -> list[Path]:
    entity_entries = _entity_entries(world)
    unique_entries = {
        (folder_name, entity_id) for folder_name, entity_id in entity_entries.values()
    }
    return sorted(
        world_root_path / folder_name / f"{_safe_filename(entity_id)}.txt"
        for folder_name, entity_id in unique_entries
    )


def _package_file_paths(world_root_path: Path, world: dict[str, object]) -> list[Path]:
    return sorted(
        world_root_path / PACKAGE_FOLDER / f"{_safe_filename(package.package_id)}.txt"
        for package in world.get("packages", [])
    )


def get_narrative_file_paths(world_root_path: str | Path, world: dict[str, object]) -> list[Path]:
    world_root_path = Path(world_root_path)
    return _entity_file_paths(world_root_path, world) + _package_file_paths(world_root_path, world)


def narrative_corpus_exists(world_root_path: str | Path, world: dict[str, object]) -> bool:
    expected_paths = get_narrative_file_paths(world_root_path, world)
    return bool(expected_paths) and all(path.exists() for path in expected_paths)


def load_narrative_corpus(world_root_path: str | Path) -> list[str]:
    world_root_path = Path(world_root_path)
    corpus_files: list[Path] = []
    for folder_name in (*WORLD_ENTITY_FOLDERS, PACKAGE_FOLDER):
        folder = world_root_path / folder_name
        if folder.exists():
            corpus_files.extend(sorted(folder.glob("*.txt")))

    return [path.read_text(encoding="utf-8") for path in corpus_files]


async def narrativize_corpus(
    world_root_path: str | Path,
    use_llm: bool = False,
) -> list[Path]:
    world_root_path = Path(world_root_path)
    world_root_path.mkdir(parents=True, exist_ok=True)

    world = load_world(world_root_path / WORLD_FILENAME)
    packages = list(world.get("packages", []))
    entity_entries = _entity_entries(world)

    world_without_packages = dict(world)
    world_without_packages["packages"] = []

    if use_llm:
        try:
            response = await _query_llm(
                pretty_print_world(world_without_packages),
                NARRATIVIZATION_PROMPT,
                NarrativeOutput,
            )
        except Exception:
            response = _fallback_world_narratives(world)
    else:
        response = _fallback_world_narratives(world)
    _save_world_model(world_root_path, response, entity_entries)

    if use_llm:
        try:
            package_response = await _query_llm(
                _format_packages(packages),
                PACKAGE_NARRATIVIZATION_PROMPT,
                PackageNarrativeOutput,
            )
        except Exception:
            package_response = _fallback_package_narratives(world)
    else:
        package_response = _fallback_package_narratives(world)
    _save_packages(world_root_path, package_response)

    return get_narrative_file_paths(world_root_path, world)


def run_narrativize_corpus(
    world_root_path: str | Path,
    use_llm: bool | None = None,
) -> list[Path]:
    if use_llm is None:
        use_llm = os.getenv("LOGISTICS_SYSTEM_USE_LLM_NARRATIVIZATION", "").lower() in {
            "1",
            "true",
            "yes",
        }

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(narrativize_corpus(world_root_path, use_llm=use_llm))

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(narrativize_corpus(world_root_path, use_llm=use_llm))
    finally:
        loop.close()

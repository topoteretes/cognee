from __future__ import annotations

import json
import re
from pathlib import Path

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.package import Package
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.post_office import (
    PostOffice,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.retailer import (
    Retailer,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.transportation import (
    Carrier,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.user import User
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.enums import (
    PostOfficeType,
    Region,
    ShippingRange,
    TransportMode,
)


DEFAULT_WORLD_PATH = Path("data").joinpath("stored_world.json")


def _resolve_world_path(path: str | Path = DEFAULT_WORLD_PATH) -> Path:
    resolved_path = Path(path)
    if resolved_path.suffix == ".json":
        return resolved_path
    return resolved_path.joinpath("stored_world.json")


def _safe_filename(key: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", key).strip("_")
    return normalized or "entity"


def _entity_entries(world: dict[str, object]) -> dict[str, tuple[Path, str]]:
    entity_entries: dict[str, tuple[Path, str]] = {}

    def add_entry(folder: str, entity_id: str, entity_name: str) -> None:
        entry = (Path(folder), entity_id)
        entity_entries[entity_id] = entry
        entity_entries[entity_name] = entry
        entity_entries[entity_name.lower()] = entry

    for retailer in world.get("retailers", []):
        add_entry("retailer", retailer.retailer_id, retailer.name)
    for user in world.get("users", []):
        add_entry("user", user.user_id, user.name)
    for post_office in world.get("post_offices", []):
        add_entry("post_office", post_office.post_office_id, post_office.name)
    for carrier in world.get("carriers", []):
        add_entry("carrier", carrier.carrier_id, carrier.company_name)

    return entity_entries


def _format_packages(packages: list[Package]) -> str:
    lines = ["Packages"]
    for package in packages:
        lines.extend(
            [
                f"  - Package ID: {package.package_id}",
                f"    Description: {package.description}",
                f"    Retailer Name: {package.retailer_name or 'unknown'}",
                f"    User Name: {package.user_name or 'unknown'}",
                f"    Weight (kg): {package.weight_kg}",
                f"    Shipping Range: {package.shipping_range.label}",
                f"    Category: {package.category.label}",
                f"    Priority: {package.priority.label}",
                f"    Insurance Status: {'insured' if package.insured else 'not insured'}",
                f"    Current State: {package.current_state.label}",
                f"    Last Known Location: {package.last_known_location}",
                f"    Current Post Office: {package.current_post_office_name or 'none'}",
                f"    Post Office Route: {', '.join(package.route_post_office_names) if package.route_post_office_names else 'none'}",
                f"    Status History: {', '.join(package.status_history) if package.status_history else 'none'}",
            ]
        )
    return "\n".join(lines)


def _serialize_retailer(retailer: Retailer) -> dict[str, object]:
    return {
        "retailer_id": retailer.retailer_id,
        "name": retailer.name,
        "region": retailer.region.label,
        "shipping_range": retailer.shipping_range.label,
        "handling_fee": retailer.handling_fee,
        "processing_days": retailer.processing_days,
        "origin_post_office_id": retailer.origin_post_office_id,
    }


def _serialize_user(user: User) -> dict[str, object]:
    return {
        "user_id": user.user_id,
        "name": user.name,
        "tier": user.tier.label,
        "region": user.region.label,
        "weekend_delivery_eligible": user.weekend_delivery_eligible,
        "default_shipping_range": user.default_shipping_range.label,
    }


def _serialize_package(package: Package) -> dict[str, object]:
    return {
        "package_id": package.package_id,
        "description": package.description,
        "weight_kg": package.weight_kg,
        "shipping_range": package.shipping_range.label,
        "category": package.category.label,
        "priority": package.priority.label,
        "retailer_id": package.retailer_id,
        "retailer_name": package.retailer_name,
        "user_id": package.user_id,
        "user_name": package.user_name,
        "insured": package.insured,
        "current_state": package.current_state.label,
        "last_known_location": package.last_known_location,
        "current_post_office_id": package.current_post_office_id,
        "current_post_office_name": package.current_post_office_name,
        "route_post_office_ids": list(package.route_post_office_ids),
        "route_post_office_names": list(package.route_post_office_names),
        "status_history": list(package.status_history),
    }


def _serialize_post_office(post_office: PostOffice) -> dict[str, object]:
    return {
        "post_office_id": post_office.post_office_id,
        "name": post_office.name,
        "office_type": post_office.office_type.label,
        "region": post_office.region.label,
        "shipping_range": post_office.shipping_range.label,
        "supports_cold_chain": post_office.supports_cold_chain,
        "supports_hazardous_materials": post_office.supports_hazardous_materials,
    }


def _serialize_carrier(carrier: Carrier) -> dict[str, object]:
    return {
        "carrier_id": carrier.carrier_id,
        "company_name": carrier.company_name,
        "region": carrier.region.label,
        "supported_modes": [mode.label for mode in carrier.supported_modes],
        "shipping_ranges": [shipping_range.label for shipping_range in carrier.shipping_ranges],
        "temperature_controlled": carrier.temperature_controlled,
        "hazardous_certified": carrier.hazardous_certified,
        "weekend_operations": carrier.weekend_operations,
        "max_weight_kg": carrier.max_weight_kg,
        "reliability_score": carrier.reliability_score,
        "base_delay_days": carrier.base_delay_days,
    }


def _deserialize_retailer(data: dict[str, object]) -> Retailer:
    return Retailer.from_labels(
        retailer_id=str(data["retailer_id"]),
        name=str(data["name"]),
        region=str(data["region"]),
        shipping_range=str(data["shipping_range"]),
        handling_fee=float(data["handling_fee"]),
        processing_days=int(data["processing_days"]),
        origin_post_office_id=str(data["origin_post_office_id"])
        if data["origin_post_office_id"] is not None
        else None,
    )


def _deserialize_user(data: dict[str, object]) -> User:
    return User.from_labels(
        user_id=str(data["user_id"]),
        name=str(data["name"]),
        tier=str(data["tier"]),
        region=str(data["region"]),
        weekend_delivery_eligible=bool(data["weekend_delivery_eligible"]),
        default_shipping_range=str(data["default_shipping_range"]),
    )


def _deserialize_package(data: dict[str, object]) -> Package:
    return Package.from_labels(
        package_id=str(data["package_id"]),
        description=str(data["description"]),
        weight_kg=float(data["weight_kg"]),
        shipping_range=str(data["shipping_range"]),
        category=str(data["category"]),
        priority=str(data["priority"]),
        retailer_id=str(data["retailer_id"]) if data.get("retailer_id") is not None else None,
        retailer_name=str(data["retailer_name"]) if data.get("retailer_name") is not None else None,
        user_id=str(data["user_id"]) if data.get("user_id") is not None else None,
        user_name=str(data["user_name"]) if data.get("user_name") is not None else None,
        insured=bool(data["insured"]),
        current_state=str(data["current_state"]),
        last_known_location=str(data["last_known_location"]),
        current_post_office_id=str(data["current_post_office_id"])
        if data["current_post_office_id"] is not None
        else None,
        current_post_office_name=str(data["current_post_office_name"])
        if data["current_post_office_name"] is not None
        else None,
        route_post_office_ids=tuple(str(value) for value in data.get("route_post_office_ids", [])),
        route_post_office_names=tuple(
            str(value) for value in data.get("route_post_office_names", [])
        ),
        status_history=tuple(str(value) for value in data.get("status_history", [])),
    )


def _deserialize_post_office(data: dict[str, object]) -> PostOffice:
    return PostOffice(
        post_office_id=str(data["post_office_id"]),
        name=str(data["name"]),
        office_type=PostOfficeType.from_label(str(data["office_type"])),
        region=Region.from_label(str(data["region"])),
        shipping_range=ShippingRange.from_label(str(data["shipping_range"])),
        supports_cold_chain=bool(data["supports_cold_chain"]),
        supports_hazardous_materials=bool(data["supports_hazardous_materials"]),
    )


def _deserialize_carrier(data: dict[str, object]) -> Carrier:
    return Carrier(
        carrier_id=str(data["carrier_id"]),
        company_name=str(data["company_name"]),
        region=Region.from_label(str(data["region"])),
        supported_modes=tuple(
            TransportMode.from_label(str(value)) for value in data["supported_modes"]
        ),
        shipping_ranges=tuple(
            ShippingRange.from_label(str(value)) for value in data["shipping_ranges"]
        ),
        temperature_controlled=bool(data["temperature_controlled"]),
        hazardous_certified=bool(data["hazardous_certified"]),
        weekend_operations=bool(data["weekend_operations"]),
        max_weight_kg=float(data["max_weight_kg"]),
        reliability_score=int(data["reliability_score"]),
        base_delay_days=int(data["base_delay_days"]),
    )


def _find_by_id(items: list[object], attr_name: str, target_id: str | None) -> object | None:
    if target_id is None:
        return None
    for item in items:
        if getattr(item, attr_name) == target_id:
            return item
    return None


def _supports_shipping_range(candidate_range: ShippingRange, package_range: ShippingRange) -> bool:
    return candidate_range.distance_score >= package_range.distance_score


def match_user_and_retailer_for_package(
    world: dict[str, object],
    package: Package,
) -> tuple[User, Retailer]:
    users = list(world.get("users", []))
    retailers = list(world.get("retailers", []))

    if not users:
        raise ValueError("World does not contain any users to match against a package.")
    if not retailers:
        raise ValueError("World does not contain any retailers to match against a package.")

    if package.user_id is not None and package.retailer_id is not None:
        matched_user = _find_by_id(users, "user_id", package.user_id)
        matched_retailer = _find_by_id(retailers, "retailer_id", package.retailer_id)
        if matched_user is not None and matched_retailer is not None:
            return matched_user, matched_retailer

    compatible_retailers = [
        retailer
        for retailer in retailers
        if _supports_shipping_range(retailer.shipping_range, package.shipping_range)
    ]
    if not compatible_retailers:
        raise ValueError(
            f"No compatible retailer found for package {package.package_id} with shipping range "
            f"{package.shipping_range.label}."
        )

    best_pair: tuple[User, Retailer] | None = None
    best_score: tuple[int, int, int] | None = None

    for user in users:
        user_supports = _supports_shipping_range(
            user.default_shipping_range, package.shipping_range
        )
        for retailer in compatible_retailers:
            score = (
                int(user_supports),
                int(user.region is retailer.region),
                retailer.shipping_range.distance_score + user.default_shipping_range.distance_score,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (user, retailer)

    if best_pair is None:
        raise ValueError(f"Could not match a user and retailer for package {package.package_id}.")

    return best_pair


def store_world(world: dict[str, object], path: str | Path = DEFAULT_WORLD_PATH) -> Path:
    retailers = list(world.get("retailers", []))
    users = list(world.get("users", []))
    packages = list(world.get("packages", []))
    post_offices = list(world.get("post_offices", []))
    carriers = list(world.get("carriers", []))

    payload = {
        "retailers": [_serialize_retailer(retailer) for retailer in retailers],
        "users": [_serialize_user(user) for user in users],
        "packages": [_serialize_package(package) for package in packages],
        "post_offices": [_serialize_post_office(post_office) for post_office in post_offices],
        "carriers": [_serialize_carrier(carrier) for carrier in carriers],
    }

    path = _resolve_world_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_world(path: str | Path = DEFAULT_WORLD_PATH) -> dict[str, object]:
    payload = json.loads(_resolve_world_path(path).read_text(encoding="utf-8"))

    retailers = [_deserialize_retailer(item) for item in payload.get("retailers", [])]
    users = [_deserialize_user(item) for item in payload.get("users", [])]
    packages = [_deserialize_package(item) for item in payload.get("packages", [])]
    post_offices = [_deserialize_post_office(item) for item in payload.get("post_offices", [])]
    carriers = [_deserialize_carrier(item) for item in payload.get("carriers", [])]

    world: dict[str, object] = {
        "retailers": retailers,
        "users": users,
        "post_offices": post_offices,
        "carriers": carriers,
    }

    if packages:
        world["packages"] = packages

    return world

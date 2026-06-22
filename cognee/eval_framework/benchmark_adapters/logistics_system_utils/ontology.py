from __future__ import annotations

import random
import json
from pathlib import Path
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.package import Package
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.retailer import (
    retailer_possible_values,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.rule_engine import (
    DeliveryRuleEngine,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.utils.utils import (
    store_world,
    load_world,
    match_user_and_retailer_for_package,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.utils.world_creation_utils import (
    _normalize_entity_count,
    _normalize_package_count,
    _seed_user,
    _pick_unique_name,
    _seed_retailer,
    _seed_post_offices,
    _origin_post_office_id_for_retailer,
    _seed_world_carriers,
    _compatible_user_retailer_pairs,
    _seed_package,
    USER_NAMES,
    RETAILER_NAMES,
    PACKAGE_DESCRIPTIONS,
    pretty_print_world,
)


def create_world(
    user_count: int = 15,
    retailer_count: int = 10,
) -> dict[str, object]:
    user_count = _normalize_entity_count(user_count)
    retailer_count = _normalize_entity_count(retailer_count)

    used_user_names: set[str] = set()
    used_retailer_names: set[str] = set()

    user = _seed_user(name=_pick_unique_name(USER_NAMES, used_user_names, "User"))
    retailer = _seed_retailer(
        shipping_range=user.default_shipping_range.label,
        name=_pick_unique_name(RETAILER_NAMES, used_retailer_names, "Retailer"),
    )
    post_offices = _seed_post_offices(user.default_shipping_range, retailer.region, user.region)
    retailer.origin_post_office_id = _origin_post_office_id_for_retailer(
        retailer.region, post_offices
    )

    users = [user]
    for _ in range(user_count - 1):
        users.append(
            _seed_user(
                name=_pick_unique_name(USER_NAMES, used_user_names, "User"),
            )
        )

    retailers = [retailer]
    for _ in range(retailer_count - 1):
        retailers.append(
            _seed_retailer(
                shipping_range=random.choice(retailer_possible_values["shipping_range"]),
                origin_post_office_id=None,
                name=_pick_unique_name(RETAILER_NAMES, used_retailer_names, "Retailer"),
            )
        )

    for retailer in retailers:
        retailer.origin_post_office_id = _origin_post_office_id_for_retailer(
            retailer.region, post_offices
        )

    carriers = _seed_world_carriers(
        retailer=retailer,
        user=user,
        shipping_range=user.default_shipping_range,
    )

    return {
        "retailers": retailers,
        "users": users,
        "post_offices": post_offices,
        "carriers": carriers,
    }


def add_packages_to_world(
    world: dict[str, object],
    package_count: int = 1,
) -> dict[str, object]:
    package_count = _normalize_package_count(package_count)

    users = list(world.get("users", []))
    retailers = list(world.get("retailers", []))
    post_offices = world["post_offices"]

    if not users:
        raise ValueError("World must contain users before packages can be created.")
    if not retailers:
        raise ValueError("World must contain retailers before packages can be created.")

    compatible_pairs = _compatible_user_retailer_pairs(users, retailers)
    if not compatible_pairs:
        raise ValueError(
            "World does not contain any user and retailer pair with a compatible shipping range."
        )

    def is_deliverable(candidate_package: Package, candidate_packages: list[Package]) -> bool:
        candidate_world = dict(world)
        candidate_world["packages"] = candidate_packages
        matched_user, matched_retailer = match_user_and_retailer_for_package(
            candidate_world,
            candidate_package,
        )
        delivery_plan = DeliveryRuleEngine().evaluate(
            retailer=matched_retailer,
            user=matched_user,
            world=candidate_world,
            package_index=len(candidate_packages) - 1,
        )
        return delivery_plan.selected_option is not None

    max_attempts_per_package = 50
    used_package_descriptions: set[str] = set()
    packages: list[Package] = []
    for _ in range(package_count):
        for _attempt in range(max_attempts_per_package):
            matched_user, matched_retailer, package_shipping_range = random.choice(compatible_pairs)
            candidate_package = _seed_package(
                post_offices=post_offices,
                shipping_range=package_shipping_range.label,
                description=_pick_unique_name(
                    PACKAGE_DESCRIPTIONS, used_package_descriptions, "Package"
                ),
                origin_region=matched_retailer.region,
                preferred_origin_post_office_id=matched_retailer.origin_post_office_id,
                retailer_id=matched_retailer.retailer_id,
                retailer_name=matched_retailer.name,
                user_id=matched_user.user_id,
                user_name=matched_user.name,
            )
            candidate_packages = [*packages, candidate_package]
            if is_deliverable(candidate_package, candidate_packages):
                packages.append(candidate_package)
                break
        else:
            raise ValueError(
                "Could not generate a deliverable logistics package after multiple attempts."
            )

    world_with_packages = dict(world)
    world_with_packages["packages"] = packages
    return world_with_packages


def validate_world_packages(world: dict[str, object]) -> None:
    packages = list(world.get("packages", []))
    for package_index, package in enumerate(packages):
        user, retailer = match_user_and_retailer_for_package(world, package)
        delivery_plan = DeliveryRuleEngine().evaluate(
            retailer=retailer,
            user=user,
            package_index=package_index,
            world=world,
        )
        if delivery_plan.selected_option is None and not package.is_received:
            raise ValueError(f"Package {package.package_id} has no deliverable transport option.")


def write_golden_answers(
    world: dict[str, object],
    world_path: Path,
) -> Path:
    validate_world_packages(world)

    golden_answers = {"golden_answers": []}
    for i in range(0, len(world["packages"])):
        user, retailer = match_user_and_retailer_for_package(world, world["packages"][i])
        delivery_plan = DeliveryRuleEngine().evaluate(
            retailer=retailer,
            user=user,
            package_index=i,
            world=world,
        )
        golden_answers["golden_answers"].append(delivery_plan.to_dict())

    golden_answers_path = world_path.joinpath("golden_answers.json")
    golden_answers_path.write_text(json.dumps(golden_answers, indent=4))
    return golden_answers_path


def build_world_and_golden_answers(
    world_path: Path,
    create_new_world: bool = False,
    user_count: int = 15,
    retailer_count: int = 10,
    package_count: int = 5,
) -> dict[str, object]:
    if create_new_world:
        world = add_packages_to_world(
            create_world(user_count=user_count, retailer_count=retailer_count),
            package_count=package_count,
        )
        store_world(world, world_path)
    else:
        world = load_world(world_path.joinpath("stored_world.json"))

    pretty_print_world(world)
    write_golden_answers(world, world_path)
    return world


def main(world_path: Path = None, create_new_world: bool = False) -> None:
    build_world_and_golden_answers(
        world_path=world_path,
        create_new_world=create_new_world,
    )


if __name__ == "__main__":
    main(world_path=Path("examples/example1"))

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.package import (
    Package,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.ontology import (
    add_packages_to_world,
    create_world,
    write_golden_answers,
)


def _build_package(
    package_id: str,
    retailer_id: str,
    retailer_name: str,
    user_id: str,
    user_name: str,
) -> Package:
    return Package.from_labels(
        package_id=package_id,
        description="medical samples",
        weight_kg=1.0,
        shipping_range="regional",
        category="standard",
        priority="standard",
        retailer_id=retailer_id,
        retailer_name=retailer_name,
        user_id=user_id,
        user_name=user_name,
        current_state="created",
    )


def test_add_packages_to_world_retries_until_package_is_deliverable():
    world = create_world(user_count=2, retailer_count=2)
    retailer = world["retailers"][0]
    user = world["users"][0]

    undeliverable_package = _build_package(
        package_id="pkg-undeliverable",
        retailer_id=retailer.retailer_id,
        retailer_name=retailer.name,
        user_id=user.user_id,
        user_name=user.name,
    )
    deliverable_package = _build_package(
        package_id="pkg-deliverable",
        retailer_id=retailer.retailer_id,
        retailer_name=retailer.name,
        user_id=user.user_id,
        user_name=user.name,
    )

    with (
        patch(
            "cognee.eval_framework.benchmark_adapters.logistics_system_utils.ontology._compatible_user_retailer_pairs",
            return_value=[(user, retailer, undeliverable_package.shipping_range)],
        ),
        patch(
            "cognee.eval_framework.benchmark_adapters.logistics_system_utils.ontology._seed_package",
            side_effect=[undeliverable_package, deliverable_package],
        ),
        patch(
            "cognee.eval_framework.benchmark_adapters.logistics_system_utils.ontology.DeliveryRuleEngine.evaluate",
            side_effect=[
                SimpleNamespace(selected_option=None),
                SimpleNamespace(selected_option=object()),
            ],
        ),
    ):
        world_with_packages = add_packages_to_world(world, package_count=1)

    assert len(world_with_packages["packages"]) == 1
    assert world_with_packages["packages"][0].package_id == "pkg-deliverable"


def test_write_golden_answers_raises_for_undeliverable_package(tmp_path: Path):
    world = create_world(user_count=2, retailer_count=2)
    retailer = world["retailers"][0]
    user = world["users"][0]
    world["packages"] = [
        _build_package(
            package_id="pkg-undeliverable",
            retailer_id=retailer.retailer_id,
            retailer_name=retailer.name,
            user_id=user.user_id,
            user_name=user.name,
        )
    ]

    with patch(
        "cognee.eval_framework.benchmark_adapters.logistics_system_utils.ontology.DeliveryRuleEngine.evaluate",
        return_value=SimpleNamespace(selected_option=None),
    ):
        with pytest.raises(
            ValueError,
            match="pkg-undeliverable has no deliverable transport option",
        ):
            write_golden_answers(world, tmp_path)

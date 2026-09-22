"""Exercise an installed experimental wheel with local GLiNER and no LLM keys.

Run from a clean environment, outside the checkout. See
docs/gliner-bundled-runtime.md for required isolation and cache settings.
This script does not install packages, launch services, or publish artifacts.
"""

import asyncio
import importlib.util
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(os.environ["COGNEE_BUNDLE_TEST_ROOT"]).resolve()
LABEL = os.environ.get("COGNEE_BUNDLE_TEST_LABEL", "smoke")
DEVICE = os.environ.get("COGNEE_GLINER_DEVICE", "cpu")
EXTERNAL_INSTALLED = os.environ.get("COGNEE_BUNDLE_EXTERNAL_INSTALLED") == "1"
OUTPUT = ROOT / f"{LABEL}-smoke.json"


def inventory():
    return {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in metadata.distributions()
    }


async def main():
    started = time.monotonic()
    evidence = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        assert ROOT.is_dir(), "Create the isolated test directory first"
        assert LABEL and all(char.isalnum() or char in "-_" for char in LABEL)
        assert os.environ.get("PYTHON_DOTENV_DISABLED") == "1"
        for key in ("HOME", "SYSTEM_ROOT_DIRECTORY", "DATA_ROOT_DIRECTORY"):
            assert Path(os.environ[key]).resolve().is_relative_to(ROOT), key
        assert os.environ.get("EMBEDDING_PROVIDER") == "fastembed"
        assert os.environ.get("CACHING") == "true"
        assert os.environ.get("AUTO_FEEDBACK") == "false"
        credentials = [
            key
            for key in os.environ
            if any(
                word in key.upper() for word in ("API_KEY", "SECRET", "CREDENTIAL", "ACCESS_TOKEN")
            )
        ]
        assert not credentials, credentials
        before = inventory()
        stack = {"torch", "gliner2", "peft", "accelerate", "transformers"}
        if EXTERNAL_INSTALLED:
            assert stack <= before.keys()
        else:
            assert not (stack & before.keys())
            assert importlib.util.find_spec("torch") is None
        evidence["installed_before_cognee_import"] = before

        # isort: off
        # Cognee activates the private runtime before any ML library import.
        import cognee
        import torch
        from gliner2 import AutoExtractor
        from cognee.modules.cognify.config import get_cognify_config, resolve_extractor
        from cognee.tasks.graph.gliner_demo.extractor import load_extractor
        # isort: on

        assert os.environ["GRAPH_EXTRACTOR"] == "gliner_demo"
        assert resolve_extractor(None, get_cognify_config()) == "gliner_demo"
        vendor = Path(cognee.__file__).parent / "_vendor_cpu"
        for name in ("torch", "gliner2", "peft", "accelerate", "transformers"):
            module = __import__(name)
            assert Path(module.__file__).is_relative_to(vendor) == (DEVICE == "cpu"), (
                name,
                module.__file__,
            )
        if DEVICE == "cpu":
            assert torch.version.cuda is None
        if platform.system() == "Linux" and DEVICE == "cpu":
            assert not torch.backends.mps.is_built()
        after = inventory()
        gpu_distributions = sorted(
            name for name in after if name.startswith(("nvidia-", "cuda-", "triton"))
        )
        if not EXTERNAL_INSTALLED:
            assert not gpu_distributions, gpu_distributions
        evidence.update(
            torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,
            torch_file=torch.__file__,
            gpu_distributions=gpu_distributions,
            model_class=f"{AutoExtractor.__module__}.{AutoExtractor.__name__}",
            mps_backend_built=torch.backends.mps.is_built(),
        )

        from pip._internal.operations.check import (
            check_package_set,
            create_package_set_from_installed,
        )

        packages, problems = create_package_set_from_installed()
        missing, conflicting = check_package_set(packages)
        assert not problems and not missing and not conflicting, (problems, missing, conflicting)
        evidence["runtime_metadata_dependency_check"] = (
            "active-runtime requirement check passed; run pip check separately for platform tags"
        )

        import litellm

        def blocked(*args, **kwargs):
            raise AssertionError("LLM completion must not run")

        async def blocked_async(*args, **kwargs):
            raise AssertionError("LLM completion must not run")

        litellm.completion = blocked
        litellm.acompletion = blocked_async
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model = load_extractor()
        expected_device = "cpu" if DEVICE == "cpu" else f"{DEVICE}:0"
        actual_devices = {str(param.device) for param in model.parameters()}
        assert actual_devices == {expected_device}, actual_devices
        result = model.extract_entities(
            "Leo Tolstoy wrote War and Peace, a novel set in Russia.",
            ["person", "book", "country"],
        )
        assert result["entities"] == {
            "person": ["Leo Tolstoy"],
            "book": ["War and Peace"],
            "country": ["Russia"],
        }, result
        evidence["extraction"] = result
        evidence["parameter_devices"] = sorted(actual_devices)
        evidence["runtime_selection"] = DEVICE
        evidence["mps_available"] = torch.backends.mps.is_available()
        evidence["mps_cpu_fallback_enabled"] = os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK", "unset"
        )
        print("EXTRACTION " + json.dumps(result), flush=True)
        OUTPUT.write_text(json.dumps(evidence, indent=2, default=str))

        cognee.config.system_root_directory(str(ROOT / f"{LABEL}-runtime/system"))
        cognee.config.data_root_directory(str(ROOT / f"{LABEL}-runtime/data"))
        remembered = await cognee.remember(
            "Marie Curie was born in Warsaw. Pierre Curie was the husband of Marie Curie. They worked in Paris.",
            dataset_name="single_wheel_cpu",
            self_improvement=False,
        )
        evidence["remember"] = str(remembered)
        recalled = await cognee.recall(
            "Where did Marie Curie work?",
            datasets=["single_wheel_cpu"],
            query_type=cognee.SearchType.CHUNKS,
            top_k=3,
        )
        assert recalled and "Paris" in str(recalled), recalled
        evidence["recall"] = recalled
        from cognee.context_global_variables import set_database_global_context_variables
        from cognee.infrastructure.databases.graph import get_graph_engine
        from cognee.modules.users.methods import get_default_user

        user = await get_default_user()
        dataset = next(
            item
            for item in await cognee.datasets.list_datasets(user)
            if item.name == "single_wheel_cpu"
        )
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            engine = await get_graph_engine()
            nodes, edges = await engine.get_graph_data()
        assert nodes and edges
        evidence["graph"] = {"nodes": len(nodes), "edges": len(edges)}
        (ROOT / f"{LABEL}-graph.json").write_text(
            json.dumps({"nodes": nodes, "edges": edges}, default=str, indent=2)
        )
        evidence["success"] = True
    except Exception as error:
        evidence["success"] = False
        evidence["error"] = repr(error)
        raise
    finally:
        evidence["seconds"] = round(time.monotonic() - started, 2)
        OUTPUT.write_text(json.dumps(evidence, indent=2, default=str))
        print(
            "RESULT "
            + json.dumps(
                {
                    key: value
                    for key, value in evidence.items()
                    if key not in {"installed_before_cognee_import", "recall", "remember"}
                },
                default=str,
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())

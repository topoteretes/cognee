"""Offline packaging/runtime tests; no Cognee import or native ML libraries needed.

Can also run with unittest discovery to avoid the main pytest conftest setup.
"""

import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from email.parser import BytesParser
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("gliner_bundle", ROOT / "tools/gliner_bundle.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class WheelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.cognee = self.wheel(
            "cognee",
            "1.6.0",
            requirements=[
                'gliner2[local]>=2.0.0; extra == "gliner"',
                'sentencepiece>=0.2; extra == "gliner"',
                "requests>=2",
            ],
            files={
                "cognee/__init__.py": b"from cognee.version import get_cognee_version\n",
                "cognee/tasks/graph/gliner_demo/extractor.py": b"""def load_extractor(model_name):
    with _load_lock:
        require_gliner2()
        extractor = AutoExtractor.from_pretrained(model_name)
        return extractor
""",
            },
        )
        self.runtime = []
        for name, version in builder.RUNTIME_VERSIONS.items():
            tag = "py3-none-any"
            requirements = []
            if name == "torch":
                version += "+cpu"
                tag = "cp312-cp312-manylinux_2_28_aarch64"
                requirements = ["sympy>=1.13"]
            elif name == "gliner2":
                requirements = ['torch>=2; extra == "local"', 'numpy>=1; extra == "local"']
            else:
                requirements = ["torch>=2"]
            self.runtime.append(self.wheel(name, version, tag, requirements))

    def wheel(self, name, version, tag="py3-none-any", requirements=(), files=None):
        path = self.root / f"{name}-{version}-{tag}.whl"
        info = f"{name}-{version}.dist-info"
        meta = f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\nRequires-Python: >=3.10\n"
        meta += "".join(f"Requires-Dist: {value}\n" for value in requirements)
        if name == "cognee":
            meta += "Provides-Extra: gliner\n"
        with ZipFile(path, "w") as wheel:
            wheel.writestr(info + "/METADATA", meta)
            wheel.writestr(info + "/WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
            wheel.writestr(info + "/RECORD", "")
            wheel.writestr(info + "/LICENSE", "Original upstream notice")
            for filename, content in (
                files or {f"{name}/__init__.py": b"# Original bytes\n"}
            ).items():
                wheel.writestr(filename, content)
        return path

    def build(self, **overrides):
        args = {
            "cognee": self.cognee,
            "runtime": self.runtime,
            "output": self.root / "dist",
            "version": None,
        }
        args.update(overrides)
        return builder.build(SimpleNamespace(**args))

    def test_wheel_metadata_extras_bytes_and_record(self):
        output = self.build()
        self.assertTrue(output.name.endswith("cp312-cp312-manylinux_2_28_aarch64.whl"))
        with ZipFile(output) as wheel:
            info = "cognee-1.6.0+local.gliner.dist-info/"
            meta = BytesParser().parsebytes(wheel.read(info + "METADATA"))
            self.assertEqual(meta["Requires-Python"], ">=3.12,<3.13")
            self.assertIn("gliner-gpu", meta.get_all("Provides-Extra"))
            requirements = [Requirement(value) for value in meta.get_all("Requires-Dist")]
            for req in requirements:
                if req.name in builder.BUNDLED:
                    self.assertFalse(req.marker.evaluate({"extra": ""}))
                    self.assertTrue(req.marker.evaluate({"extra": "gliner-gpu"}))
            base = {req.name for req in requirements if not req.marker}
            self.assertTrue({"numpy", "sympy", "sentencepiece", "requests"} <= base)
            manifest = json.loads(wheel.read(builder.PREFIX + "BUNDLED_MANIFEST.json"))
            self.assertEqual(len(manifest), 5)
            for source in self.runtime:
                with ZipFile(source) as original:
                    for member in original.namelist():
                        self.assertEqual(wheel.read(builder.PREFIX + member), original.read(member))
            record = info + "RECORD"
            rows = list(csv.reader(io.StringIO(wheel.read(record).decode())))
            self.assertEqual({row[0] for row in rows}, set(wheel.namelist()))
            for name, digest, size in rows:
                if name != record:
                    data = wheel.read(name)
                    self.assertEqual(digest, "sha256=" + builder.digest(data))
                    self.assertEqual(int(size), len(data))
            extractor = wheel.read("cognee/tasks/graph/gliner_demo/extractor.py").decode()
            self.assertLess(
                extractor.index("device = extraction_device()"), extractor.index("with _load_lock:")
            )
            self.assertIn(".to(device).eval()", extractor)
            compile(extractor, "extractor.py", "exec")
            compile(wheel.read("cognee/_cpu_runtime.py"), "runtime.py", "exec")

    def test_does_not_overwrite_an_existing_artifact(self):
        output = self.build()
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(output.read_bytes(), original)

    def test_rejects_non_cpu_linux_input(self):
        self.runtime[0] = self.wheel("torch", "2.14.0", "cp312-cp312-manylinux_2_28_aarch64")
        with self.assertRaisesRegex(ValueError, "official.*cpu"):
            self.build()

    def test_rejects_untested_platform(self):
        self.runtime[0] = self.wheel("torch", "2.14.0+cpu", "cp312-cp312-manylinux_2_28_x86_64")
        with self.assertRaisesRegex(ValueError, "Unsupported target"):
            self.build()

    def test_rejects_wrong_runtime_version(self):
        self.runtime[0] = self.wheel("torch", "2.13.0+cpu", "cp312-cp312-manylinux_2_28_aarch64")
        with self.assertRaisesRegex(ValueError, "Expected torch==2.14.0"):
            self.build()

    def test_rejects_duplicate_or_missing_inputs(self):
        with self.assertRaisesRegex(ValueError, "Need exactly"):
            self.build(runtime=self.runtime + [self.runtime[0]])
        with self.assertRaisesRegex(ValueError, "Need exactly"):
            self.build(runtime=self.runtime[:-1])

    def test_rejects_release_version(self):
        with self.assertRaisesRegex(ValueError, "local version suffix"):
            self.build(version="1.6.0")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.vendor = self.root / "_vendor_cpu"
        self.vendor.mkdir()
        (self.vendor / "BUNDLED_MANIFEST.json").write_text("[]")
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        # These are control-flow unit doubles, not claims of real GPU inference.
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in (*builder.BUNDLED, "functorch", "torchgen"):
            sys.modules.pop(name, None)
        self.path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.path))

    def runtime(self, device="cpu"):
        os.environ["COGNEE_GLINER_DEVICE"] = device
        module = ModuleType("test_runtime")
        module.__file__ = str(self.root / "_cpu_runtime.py")
        # Execute only the repository-owned runtime template, never input wheel code.
        exec(compile(builder.ACTIVATOR, module.__file__, "exec"), module.__dict__)  # noqa: S102
        return module

    def test_cpu_activation_is_idempotent(self):
        runtime = self.runtime()
        runtime.activate()
        runtime.activate()
        self.assertEqual(sys.path[0], str(self.vendor))
        self.assertEqual(sys.path.count(str(self.vendor)), 1)

    def test_cpu_rejects_external_preimport(self):
        sys.modules["torch"] = SimpleNamespace(__file__="/external/torch/__init__.py")
        with self.assertRaisesRegex(RuntimeError, "Already imported torch conflicts"):
            self.runtime().activate()

    def test_missing_bundle_fails(self):
        (self.vendor / "BUNDLED_MANIFEST.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing or incomplete"):
            self.runtime().activate()

    def test_gpu_uses_external_versions_without_vendor_path(self):
        runtime = self.runtime("cuda")
        with patch.object(runtime, "version", side_effect=builder.RUNTIME_VERSIONS.__getitem__):
            runtime.activate()
        self.assertNotIn(str(self.vendor), sys.path)

    def test_gpu_rejects_preimported_bundle(self):
        sys.modules["torch"] = SimpleNamespace(__file__=str(self.vendor / "torch/__init__.py"))
        with self.assertRaisesRegex(RuntimeError, "Already imported torch conflicts"):
            self.runtime("cuda").activate()

    def test_missing_extra_fails(self):
        runtime = self.runtime("cuda")
        with (
            patch.object(runtime, "version", side_effect=runtime.PackageNotFoundError),
            self.assertRaisesRegex(RuntimeError, "gliner-gpu"),
        ):
            runtime.activate()

    def test_wrong_external_version_fails(self):
        runtime = self.runtime("cuda")
        with (
            patch.object(runtime, "version", return_value="0.0.0"),
            self.assertRaisesRegex(RuntimeError, "requires torch=="),
        ):
            runtime.activate()

    def test_invalid_device_fails(self):
        with self.assertRaisesRegex(RuntimeError, "must be cpu, cuda, or mps"):
            self.runtime("auto").activate()

    def test_device_change_fails(self):
        runtime = self.runtime()
        os.environ["COGNEE_GLINER_DEVICE"] = "cuda"
        with self.assertRaisesRegex(RuntimeError, "before importing Cognee"):
            runtime.extraction_device()

    def test_unavailable_cuda_fails_without_cpu_fallback(self):
        runtime = self.runtime("cuda")
        sys.modules["torch"] = SimpleNamespace(
            version=SimpleNamespace(cuda="13.0"), cuda=SimpleNamespace(is_available=lambda: False)
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            runtime.extraction_device()

    def test_available_cuda_control_flow(self):
        runtime = self.runtime("cuda")
        sys.modules["torch"] = SimpleNamespace(
            version=SimpleNamespace(cuda="13.0"), cuda=SimpleNamespace(is_available=lambda: True)
        )
        self.assertEqual(runtime.extraction_device(), "cuda")

    def test_unavailable_mps_fails_without_cpu_fallback(self):
        runtime = self.runtime("mps")
        sys.modules["torch"] = SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        )
        with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
            runtime.extraction_device()


if __name__ == "__main__":
    unittest.main()

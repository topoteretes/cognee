"""Build an experimental Cognee wheel with bundled CPU GLiNER and a GPU extra.

Inputs are existing wheels. No downloads, installs, uploads, or publishing.
This is an opt-in local build tool, not a release entry point. See
docs/gliner-bundled-runtime.md for the tested inputs and remaining limits.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

RUNTIME_VERSIONS = {
    "torch": "2.14.0",
    "gliner2": "2.0.0",
    "transformers": "4.57.6",
    "accelerate": "1.15.0",
    "peft": "0.21.0",
}
BUNDLED = set(RUNTIME_VERSIONS)
PREFIX = "cognee/_vendor_cpu/"
SUPPORTED_TAGS = {
    "cp312-cp312-manylinux_2_28_aarch64",
    "cp312-cp312-macosx_14_0_arm64",
}

ACTIVATOR = '''"""Select one GLiNER runtime before imports; never switch within a process."""
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

DEVICE = os.environ.get("COGNEE_GLINER_DEVICE", "cpu")
VERSIONS = __RUNTIME_VERSIONS__


def activate():
    if DEVICE not in {"cpu", "cuda", "mps"}:
        raise RuntimeError("COGNEE_GLINER_DEVICE must be cpu, cuda, or mps")
    root = Path(__file__).resolve().parent / "_vendor_cpu"
    for name in ("torch", "functorch", "torchgen", "gliner2", "transformers", "accelerate", "peft"):
        module = sys.modules.get(name)
        if module is not None:
            path = getattr(module, "__file__", None)
            bundled = bool(path and Path(path).resolve().is_relative_to(root))
            if not path or bundled != (DEVICE == "cpu"):
                raise RuntimeError(
                    f"Already imported {name} conflicts with COGNEE_GLINER_DEVICE={DEVICE}. "
                    "Select the device before imports and use a fresh process."
                )
    if DEVICE == "cpu":
        if not (root / "BUNDLED_MANIFEST.json").is_file():
            raise RuntimeError("Cognee's bundled CPU runtime is missing or incomplete")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return
    if str(root) in sys.path:
        raise RuntimeError("The bundled CPU runtime is already active; use a fresh process")
    for name, expected in VERSIONS.items():
        try:
            actual = version(name)
        except PackageNotFoundError as error:
            raise RuntimeError('GPU mode requires pip install "cognee[gliner-gpu]"') from error
        if Version(actual).base_version != expected:
            raise RuntimeError(f"GPU runtime requires {name}=={expected}; found {actual}")


def extraction_device():
    """Validate requested hardware before downloading/loading a model."""
    if os.environ.get("COGNEE_GLINER_DEVICE", "cpu") != DEVICE:
        raise RuntimeError("Set COGNEE_GLINER_DEVICE before importing Cognee; restart the process")
    import torch
    if DEVICE == "cuda" and (torch.version.cuda is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but a CUDA-enabled Torch and usable NVIDIA GPU are required")
    if DEVICE == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but a usable Apple GPU is required")
    return DEVICE
'''.replace("__RUNTIME_VERSIONS__", repr(RUNTIME_VERSIONS))


def digest(data):
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def file_digest(path):
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def wheel_metadata(path):
    with ZipFile(path) as wheel:
        names = [
            name
            for name in wheel.namelist()
            if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise ValueError(f"Expected one distribution: {path}")
        return names[0].split("/")[0], BytesParser(policy=policy.compat32).parsebytes(
            wheel.read(names[0])
        )


def runtime_requirements(meta):
    """Promote GLiNER's local extra and include each bundled project's base dependencies."""
    for text in meta.get_all("Requires-Dist", []):
        req = Requirement(text)
        if req.marker and "extra" in str(req.marker):
            if meta["Name"] == "gliner2" and str(req.marker) == 'extra == "local"':
                req.marker = None
            else:
                continue
        if canonicalize_name(req.name) not in BUNDLED:
            yield str(req)


def build(args):
    """Transform local input wheels without fetching or installing dependencies."""
    base_info, meta = wheel_metadata(args.cognee)
    if canonicalize_name(meta["Name"]) != "cognee":
        raise ValueError("The base wheel must be Cognee")
    inputs = {canonicalize_name(wheel_metadata(path)[1]["Name"]): path for path in args.runtime}
    if set(inputs) != BUNDLED or len(args.runtime) != len(BUNDLED):
        raise ValueError(f"Need exactly {sorted(BUNDLED)}, received {sorted(inputs)}")
    for name, path in inputs.items():
        _, runtime_meta = wheel_metadata(path)
        actual = runtime_meta["Version"]
        expected = RUNTIME_VERSIONS[name]
        if Version(actual).base_version != expected:
            raise ValueError(f"Expected {name}=={expected}, received {actual}")
        parsed_name, parsed_version, _, _ = parse_wheel_filename(path.name)
        if parsed_name != name or parsed_version != Version(actual):
            raise ValueError(f"Wheel filename and metadata disagree: {path.name}")
    _, torch_version, _, tags = parse_wheel_filename(inputs["torch"].name)
    if len(tags) != 1:
        raise ValueError("Prototype expects one Torch wheel tag")
    tag = str(next(iter(tags)))
    if tag not in SUPPORTED_TAGS:
        raise ValueError(f"Unsupported target {tag}; tested targets: {sorted(SUPPORTED_TAGS)}")
    if "linux" in tag and "+cpu" not in str(torch_version):
        raise ValueError("Linux must use the official +cpu wheel")

    version = Version(args.version or f"{Version(meta['Version']).public}+local.gliner")
    if not version.local:
        raise ValueError("Experimental output must have a local version suffix")

    requirements = []
    for text in meta.get_all("Requires-Dist", []):
        req = Requirement(text)
        if canonicalize_name(req.name) in BUNDLED:
            if req.marker and str(req.marker) == 'extra == "gliner"':
                continue
            # Other extras remain unchanged and are explicitly outside this experiment.
        elif req.marker and str(req.marker) == 'extra == "gliner"':
            req.marker = None
        requirements.append(str(req))
    manifest = []
    for name, path in sorted(inputs.items()):
        _, runtime_meta = wheel_metadata(path)
        requirements.extend(runtime_requirements(runtime_meta))
        manifest.append(
            {
                "name": name,
                "version": runtime_meta["Version"],
                "wheel": path.name,
                "sha256": file_digest(path),
            }
        )

    # The GPU extra installs one complete external stack; it never mixes with
    # the private CPU stack. Extras cannot remove files already in the wheel.
    for name, pinned in RUNTIME_VERSIONS.items():
        extra = "[local]" if name == "gliner2" else ""
        requirements.append(f'{name}{extra}=={pinned}; extra == "gliner-gpu"')
    meta["Provides-Extra"] = "gliner-gpu"

    del meta["Requires-Dist"]
    for requirement in sorted(set(requirements)):
        meta["Requires-Dist"] = requirement
    meta.replace_header("Version", str(version))
    meta.replace_header("Requires-Python", ">=3.12,<3.13")
    new_info = f"cognee-{version}.dist-info"
    destination = args.output / f"cognee-{version}-{tag}.whl"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = set()
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=6) as result:

        def write(name, data, original=None):
            if name in paths:
                raise ValueError(f"Duplicate output member: {name}")
            paths.add(name)
            info = ZipInfo(name, (2026, 9, 22, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = original.external_attr if original else 0o644 << 16
            result.writestr(info, data)
            rows.append((name, "sha256=" + digest(data), str(len(data))))

        with ZipFile(args.cognee) as source:
            for info in source.infolist():
                name = info.filename
                if info.is_dir() or name == base_info + "/RECORD":
                    continue
                data = source.read(info)
                if name == base_info + "/METADATA":
                    data = meta.as_bytes(policy=policy.compat32.clone(max_line_length=0))
                elif name == base_info + "/WHEEL":
                    data = (
                        "Wheel-Version: 1.0\nGenerator: cognee-local-cpu-experiment\n"
                        f"Root-Is-Purelib: false\nTag: {tag}\n"
                    ).encode()
                elif name == "cognee/__init__.py":
                    text = data.decode()
                    marker = "from cognee.version import get_cognee_version"
                    if text.count(marker) != 1:
                        raise ValueError("Unexpected Cognee initialization code")
                    text = text.replace(
                        marker,
                        "from cognee._cpu_runtime import activate\nactivate()\n\n" + marker,
                        1,
                    )
                    data = text.encode()
                elif name == "cognee/tasks/graph/gliner_demo/extractor.py":
                    text = data.decode()
                    marker = "        extractor = AutoExtractor.from_pretrained(model_name)"
                    if text.count(marker) != 1:
                        raise ValueError("Unexpected GLiNER extractor initialization")
                    # Check even cached models: changing device mid-process is an error.
                    cache_marker = "    with _load_lock:"
                    if text.count(cache_marker) != 1:
                        raise ValueError("Unexpected GLiNER extractor cache")
                    text = text.replace(
                        cache_marker,
                        "    from cognee._cpu_runtime import extraction_device\n"
                        "    device = extraction_device()\n\n" + cache_marker,
                        1,
                    )
                    text = text.replace(marker, marker + ".to(device).eval()", 1)
                    data = text.encode()
                write(name.replace(base_info, new_info, 1), data, info)

        for name, path in sorted(inputs.items()):
            with ZipFile(path) as source:
                for info in source.infolist():
                    if info.is_dir():
                        continue
                    if ".data/" in info.filename:
                        raise ValueError(f"Unimplemented wheel relocation: {info.filename}")
                    # Keep original code, native libraries, metadata, licenses and notices intact.
                    write(PREFIX + info.filename, source.read(info), info)
        write("cognee/_cpu_runtime.py", ACTIVATOR.encode())
        write(PREFIX + "BUNDLED_MANIFEST.json", json.dumps(manifest, indent=2).encode())
        record = new_info + "/RECORD"
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerows(rows + [(record, "", "")])
        result.writestr(record, buffer.getvalue())

    report = {
        "wheel": str(destination),
        "bytes": destination.stat().st_size,
        "tag": tag,
        "bundled": manifest,
        "external_requirements": sorted(set(requirements)),
        "source_cognee": str(args.cognee),
        "vendored_code_modified": False,
        "prototype_limitations": [
            "CPython 3.12 only",
            "GPU requires explicit pre-import device selection",
            "Other Cognee extras are untested",
        ],
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("wheel", "bytes", "tag")}), flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cognee", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--version", help="Local output version; default: source version +local.gliner"
    )
    build(parser.parse_args())

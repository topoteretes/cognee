# Experimental bundled CPU GLiNER with an optional GPU runtime

This is an **opt-in local wheel-building prototype**, not a release change.
Normal `uv build`, source installs, `pyproject.toml`, `uv.lock`, Dockerfiles and
release workflows are unchanged. No wheels or model weights are checked in.
Do not publish these experimental artifacts or connect this tool to a release
workflow until the remaining validation below is complete.

## Installation and runtime behavior

For a wheel produced by `tools/gliner_bundle.py`:

| Installation / setting | Selected runtime |
| --- | --- |
| Plain wheel installation; device unset or `cpu` | Bundled CPU Torch and GLiNER |
| Wheel installed with `[gliner-gpu]`; device `cuda` | External Torch/GLiNER stack, NVIDIA CUDA |
| Wheel installed with `[gliner-gpu]`; device `mps` | External Torch/GLiNER stack, Apple GPU |
| Extra installed, but device unset or `cpu` | Still the bundled CPU runtime |

Set `COGNEE_GLINER_DEVICE` **before importing Cognee**, in the process environment
or with `os.environ`. Also set `GRAPH_EXTRACTOR=gliner_demo` to select GLiNER
extraction. Device selection does not change embedding configuration.

The GPU extra adds dependencies; it does not remove the CPU files in the wheel.
GPU users therefore retain both installations. Only one runtime is active in a
process. Import Cognee before ML libraries in CPU mode. Incompatible pre-imported
libraries, missing extras, unavailable requested hardware and device changes
after import raise errors; there is no automatic CPU fallback from GPU mode.

## Local build

Use Python 3.12 with `packaging` installed. Supply a Cognee wheel and these five
trusted, local upstream wheels:

| Project | Version |
| --- | --- |
| Torch, Linux ARM64 | `2.14.0+cpu`, `cp312-cp312-manylinux_2_28_aarch64` |
| Torch, Mac ARM64 | `2.14.0`, `cp312-cp312-macosx_14_0_arm64` |
| GLiNER2 | `2.0.0` |
| Transformers | `4.57.6` |
| PEFT | `0.21.0` |
| Accelerate | `1.15.0` |

Choose the Torch wheel for the target platform; the other four are pure-Python
wheels. The builder deliberately rejects untested targets and mismatched versions.

```sh
uv build --wheel --out-dir /path/to/base-wheels
python tools/gliner_bundle.py \
  --cognee /path/to/base-wheels/cognee-1.6.0-py3-none-any.whl \
  --runtime /path/to/runtime-wheels/*.whl \
  --output /path/to/experimental-wheels

python tools/verify_gliner_bundle.py /path/to/experimental-wheels/cognee-*.whl \
  --runtime /path/to/runtime-wheels/*.whl
```

The runtime input directory must contain exactly the five selected wheels.
The builder never downloads, installs or publishes anything, and refuses to
overwrite an existing wheel. Output versions have a local suffix (by default,
`1.6.0+local.gliner` for a 1.6.0 input). Build and verify each target separately.

Install a matching artifact in a fresh virtual environment using ordinary pip:

```sh
python -m pip install /path/to/experimental-wheels/cognee-...whl
python -m pip check

# Optional GPU dependencies; choose the device before starting the application.
python -m pip install '/path/to/experimental-wheels/cognee-...whl[gliner-gpu]'
```

No custom Torch index, renamed dependency, `--no-deps` installation or
dependency override is required by the consumer of the resulting wheel.

## What the builder changes

- Preserves all five upstream runtimes byte-for-byte, including their original
  metadata and notices, beneath `cognee/_vendor_cpu/`.
- Adds their external base dependencies to Cognee's wheel metadata and retains
  unrelated extras. Dependencies provided inside the bundle do not request a
  second ordinary Torch installation. The `gliner` extra is supplied by the bundle.
- Adds the `gliner-gpu` extra for a complete, separately installed runtime stack.
- Adds an early runtime selector and explicit device placement in Cognee's
  GLiNER model loader. Upstream runtime source and binaries are not patched.
- Emits a native CPython 3.12 wheel with one top-level distribution, regenerated
  `RECORD`, an input-hash manifest and an adjacent build report.

The verifier compares every bundled member against the input wheels and checks
the complete outer `RECORD`. Ordinary pip inventory does not expose the private
runtime before Cognee activates it; dependency/security inventories must include
the bundled-component manifest. Updating bundled libraries requires rebuilding
the Cognee wheel, not merely upgrading external Torch.

## Tests

Fast offline tests require only Python and `packaging`; they do not import Cognee,
download models, run native Torch, or require service credentials:

```sh
python -m unittest discover -s cognee/tests/unit/packaging -p test_gliner_bundle.py -v
```

`tools/smoke_gliner_bundle.py` exercises the installed artifact with real GLiNER,
Cognee `remember` and `recall(SearchType.CHUNKS)`. Run outside the checkout, in a
clean process environment with **no API credentials**, and provide:

- `COGNEE_BUNDLE_TEST_ROOT`: a newly created temporary directory.
- `HOME`, `SYSTEM_ROOT_DIRECTORY`, `DATA_ROOT_DIRECTORY`: paths beneath that root.
- `COGNEE_BUNDLE_TEST_LABEL`: optional unique output label, e.g. `cpu` or `mps`.
- `COGNEE_BUNDLE_EXTERNAL_INSTALLED=1` only when the GPU extra is installed.
- `PYTHON_DOTENV_DISABLED=1`, `GRAPH_EXTRACTOR=gliner_demo`, `CACHING=true`,
  `AUTO_FEEDBACK=false`, `EMBEDDING_PROVIDER=fastembed`,
  `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, `EMBEDDING_DIMENSIONS=384`.
- For offline inference: pre-populated model/embedding caches, `HF_HUB_CACHE`,
  `FASTEMBED_CACHE_PATH`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and Cognee's normal Ladybug JSON extension
  installed in the isolated HOME. Model weights are not included in the wheel.
- For an offline run without provider connection tests: `COGNEE_SKIP_CONNECTION_TEST=true`.
  Disable telemetry with `TELEMETRY_DISABLED=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
  and `LITELLM_LOCAL_MODEL_COST_MAP=True`.

Use `env -i` or an explicitly constructed subprocess environment; merely
unsetting one API key does not ensure a credential-free test. Do not enable
`PYTORCH_ENABLE_MPS_FALLBACK` for GPU validation. The smoke script records device
placement, extraction, graph counts and recall in the test directory and guards
LLM completion calls. Run `pip check` separately: the script's runtime requirement
check does not replace pip's platform-tag validation.

Local prototype tests on Python 3.12 / ARM64 demonstrated:

- Fresh plain installation and `pip check` on Mac and Linux without separately
  installed Torch, NVIDIA, CUDA or Triton distributions.
- Real CPU inference on both platforms, real Apple-GPU inference on `mps:0`,
  and continued CPU inference after installing the optional GPU runtime.
- Expected extraction of `Leo Tolstoy`, `War and Peace` and `Russia`.
- A separate Marie Curie sample produced 9 nodes / 14 edges and recalled the
  source text containing “They worked in Paris.” No LLM API keys were used.
- External Linux Torch `2.14.0+cu130` imported and reported CUDA 13.0; a CUDA
  request failed clearly on the test host without an NVIDIA GPU.

## Remaining validation before release

- Actual NVIDIA GPU inference is **not tested**. Apple MPS results do not prove
  CUDA driver/hardware compatibility. AMD/ROCm is not implemented.
- The tested upstream `nvidia-cusparselt-cu13==0.8.1` ARM64 wheel installs but
  fails `pip check`: its filename uses `manylinux2014_aarch64`, while its internal
  tag is `manylinux2014_sbsa`. This is upstream metadata, not a Cognee bundle
  conflict. It is retained unchanged, and that packaging check remains failed.
- The supported build targets are only the two listed ARM64 / Python 3.12
  combinations. Linux requires glibc, not Alpine/musl. Windows, x86_64, other
  Python versions and combinations with other Cognee extras still need testing.
- The official Mac Torch wheel includes MPS support; default mode forces CPU
  inference. It is CUDA-free, not a custom Mac binary stripped of MPS.
- Wheels are approximately 151 MB (Mac) / 183 MB (Linux), exceeding PyPI's default
  100 MB file limit. No quota change or artifact upload is part of this prototype.
- Release integration must prevent unsupported platforms from silently receiving
  an unbundled artifact if a CPU-by-default guarantee is advertised. This PR
  intentionally does not change existing release behavior or claim that guarantee
  for current public Cognee installations.

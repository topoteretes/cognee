#!/usr/bin/env bash
# Install one workspace package, and the workspace packages it depends on, into the
# cognee-free venv `.venv-clean` from the local directories, in dependency order:
# core first, then connectors/google when the package depends on it, then the
# package itself. The order comes from scripts/packages.py (the same records as
# `packages.py --json`). Third-party dependencies come from PyPI as usual; cognee is
# never installed, and the script fails if the resolution pulled it in.
#
# Create the venv first (the script creates nothing itself):
#   uv venv .venv-clean
#   bash scripts/clean_install.sh connectors/gmail
#   CLEAN_VENV=.venv-other bash scripts/clean_install.sh core
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/clean_install.sh <package dir>" >&2
  exit 2
fi

PACKAGE_DIR="${1%/}"
VENV="${CLEAN_VENV:-.venv-clean}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [ -x "${VENV}/bin/python" ]; then
  PYTHON="${VENV}/bin/python"
elif [ -x "${VENV}/Scripts/python.exe" ]; then
  PYTHON="${VENV}/Scripts/python.exe"
else
  echo "No virtual environment at ${VENV}; create it first with: uv venv ${VENV}" >&2
  exit 1
fi

if [ ! -f "${PACKAGE_DIR}/pyproject.toml" ]; then
  echo "${PACKAGE_DIR} is not a workspace package (no pyproject.toml)" >&2
  exit 1
fi

INSTALLED=""
while IFS= read -r directory; do
  [ -n "${directory}" ] || continue
  echo "==> Installing ./${directory} into ${VENV}"
  uv pip install --python "${PYTHON}" "./${directory}"
  INSTALLED="${INSTALLED}${INSTALLED:+, }${directory}"
done < <("${PYTHON}" scripts/packages.py --install-order "${PACKAGE_DIR}")

if "${PYTHON}" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("cognee") is None else 1)'; then
  echo "cognee is not installed in ${VENV}, as required."
else
  echo "ERROR: cognee was installed into ${VENV}: a package declares it as a runtime dependency (docs/ARCHITECTURE.md, rule 1)." >&2
  exit 1
fi

echo "Installed from the checkout, in order: ${INSTALLED}"

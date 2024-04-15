.PHONY: install-poetry build-library-prerelease has-poetry dev lint test test-common reset-test-storage recreate-compiled-deps build-library-prerelease publish-library

PYV=$(shell python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
.SILENT:has-poetry

# read version from package
# AUTV=$(shell cd cognee && python3 -c "from __version__ import __version__;print(__version__)")

# NAME   := topoteretes/cognee
# TAG    := $(shell git log -1 --pretty=%h)
# IMG    := ${NAME}:${TAG}
# LATEST := ${NAME}:latest${VERSION_SUFFIX}
# VERSION := ${AUTV}${VERSION_SUFFIX}
# VERSION_MM := ${AUTVMINMAJ}${VERSION_SUFFIX}

help:
	@echo "make"
	@echo "		install-poetry"
	@echo "			installs newest poetry version"
	@echo "		dev"
	@echo "			prepares development env"
	@echo "		lint"
	@echo "			runs flake and mypy"
	@echo "		test"
	@echo "			tests all the components including destinations"
	@echo "		test-load-local"
	@echo "			tests all components using local destinations"
	@echo "		test-common"
	@echo "			tests common components"
	@echo "		lint-and-test-snippets"
	@echo "			tests and lints snippets and examples in docs"
	@echo "		build-library"
	@echo "			makes dev and then builds cognee package for distribution"
	@echo "		publish-library"
	@echo "			builds library and then publishes it to pypi"

install-poetry:
ifneq ($(VIRTUAL_ENV),)
	$(error you cannot be under virtual environment $(VIRTUAL_ENV))
endif
	curl -sSL https://install.python-poetry.org | python3 -

has-poetry:
	poetry --version

dev: has-poetry
	poetry install --all-extras

lint:
	./tools/check-package.sh
	poetry run python ./tools/check-lockfile.py
	poetry run mypy --config-file mypy.ini cognee
	poetry run flake8 --max-line-length=200 cognee
	poetry run black cognee docs tests --diff --extend-exclude=".*syntax_error.py"
	# poetry run isort ./ --diff
	# $(MAKE) lint-security

format:
	poetry run black cognee docs tests --exclude=".*syntax_error.py|\.venv.*|_storage/.*"
	# poetry run isort ./

lint-and-test-snippets:
	cd docs/tools && poetry run python check_embedded_snippets.py full
	poetry run mypy --config-file mypy.ini docs/website docs/examples docs/tools --exclude docs/tools/lint_setup --exclude docs/website/docs_processed
	poetry run flake8 --max-line-length=200 docs/website docs/examples docs/tools
	cd docs/website/docs && poetry run pytest --ignore=node_modules

lint-and-test-examples:
	poetry run mypy --config-file mypy.ini docs/examples
	poetry run flake8 --max-line-length=200 docs/examples
	cd docs/tools && poetry run python prepare_examples_tests.py
	cd docs/examples && poetry run pytest


test-examples:
	cd docs/examples && poetry run pytest

lint-security:
	poetry run bandit -r cognee/ -n 3 -l

test:
	(set -a && . tests/.env && poetry run pytest tests)


build-library: dev
	poetry version
	poetry build

publish-library: build-library
	poetry publish



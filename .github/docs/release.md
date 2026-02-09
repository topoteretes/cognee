## How to make a release

### Dev release

#### Prepare release
1. Set the project version that will be released in [pyproject.toml](https://github.com/topoteretes/cognee/blob/dev/pyproject.toml#L4)
2. Update `uv.lock` with `uv lock` lock command
3. Create a PR with the changes mentioned above to `dev` and merge it.

#### Perform Release
1. Go to [Release action](https://github.com/topoteretes/cognee/actions/workflows/release.yml)
2. Select `dev` branch and run the workflow.
3. Watch the logs and make sure that everything goes well

### Main release

#### Prepare release
1. Set the project version that will be released in [pyproject.toml](https://github.com/topoteretes/cognee/blob/dev/pyproject.toml#L4)
2. Update `uv.lock` with `uv lock` lock command
3. Create a PR with the changes mentioned above to `dev` and merge it.

#### Perform Release
1. Go to [Release action](https://github.com/topoteretes/cognee/actions/workflows/release.yml)
2. Select `main` branch and run the workflow.
3. Watch the logs and make sure that everything goes well

### Release validation

1. Make sure that the correct image is published to [Docker Hub](https://hub.docker.com/r/cognee/cognee)
2. Python package is published to [PyPi](https://pypi.org/project/cognee/)
3. Find the created github release in [GitHub releases](https://github.com/topoteretes/cognee/releases). Edit/prettify the release notes if required.

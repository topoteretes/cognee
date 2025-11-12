# Docker Hub CI/CD Setup Documentation

This document describes the automated Docker image build and publishing system for the Cognee fork, covering both `cognee-mcp` and `cognee-frontend` components.

## Overview

The CI/CD pipelines automatically build multi-platform Docker images and push them to Docker Hub whenever changes are pushed to the main branch or version tags are created. The workflows are designed to run only on the fork repository (Varming73/cognee), not on pull requests to the upstream repository.

## Docker Hub Repositories

- **cognee-mcp**: [lvarming/cognee-mcp](https://hub.docker.com/r/lvarming/cognee-mcp)
- **cognee-frontend**: [lvarming/cognee-frontend](https://hub.docker.com/r/lvarming/cognee-frontend)

## Workflows

### 1. cognee-mcp Docker Build
**File**: `.github/workflows/cognee-mcp-docker.yml`

Builds the Cognee MCP server image from the `cognee-mcp/` directory.

#### Triggers
- Push to `main` branch (when files in `cognee-mcp/`, `alembic/`, or `alembic.ini` change)
- Version tags matching `v*` (e.g., `v0.4.0`, `v1.0.0`)
- Manual workflow dispatch via GitHub Actions UI

#### Version Extraction
The workflow extracts the version number from `cognee-mcp/pyproject.toml`:
```toml
[project]
name = "cognee-mcp"
version = "0.4.0"
```

### 2. cognee-frontend Docker Build
**File**: `.github/workflows/cognee-frontend-docker.yml`

Builds the Next.js frontend application image from the `cognee-frontend/` directory.

#### Triggers
- Push to `main` branch (when files in `cognee-frontend/` change)
- Version tags matching `v*` (e.g., `v1.0.0`)
- Manual workflow dispatch via GitHub Actions UI

#### Version Extraction
The workflow extracts the version number from `cognee-frontend/package.json`:
```json
{
  "name": "cognee-frontend",
  "version": "1.0.0"
}
```

## Tagging Strategy

### Automatic Tags (Push to main)
When you push to the `main` branch, the following tags are created:
- `<version>` - Version from project file (e.g., `0.4.0` or `1.0.0`)
- `<version>-<short-sha>` - Version with commit SHA (e.g., `0.4.0-abc1234`)
- `latest` - Always points to the most recent build
- `<short-sha>` - Commit SHA only (e.g., `abc1234`)

### Git Tag Push
When you push a version tag (e.g., `git tag v0.5.0 && git push origin v0.5.0`):
- `<tag-version>` - The semantic version without 'v' (e.g., `0.5.0`)
- `latest` - Updated to this version
- `<short-sha>` - Commit SHA for this tag

### Manual Workflow Dispatch
When triggered manually with a custom tag:
- `<custom-tag>` - Your specified tag
- `latest` - Also updated (unless custom tag is "latest")

## Multi-Platform Builds

Both workflows build images for multiple architectures:
- `linux/amd64` - x86_64 servers and desktops
- `linux/arm64` - ARM-based systems (Apple Silicon, Raspberry Pi, AWS Graviton)

This is achieved using Docker Buildx with QEMU for cross-platform emulation.

## GitHub Secrets Configuration

To enable the workflows, you must configure the following secrets in your GitHub repository:

### Required Secrets

1. **DOCKERHUB_USERNAME**
   - Your Docker Hub username
   - Example: `lvarming`

2. **DOCKERHUB_TOKEN**
   - Docker Hub access token (NOT your password)
   - Create at: https://hub.docker.com/settings/security

### Setting Up Secrets

1. Go to your GitHub repository: https://github.com/Varming73/cognee
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add each secret with the exact names shown above

### Creating a Docker Hub Access Token

1. Log in to Docker Hub: https://hub.docker.com
2. Go to Account Settings → Security
3. Click "New Access Token"
4. Give it a descriptive name (e.g., "GitHub Actions - Cognee")
5. Set permissions to "Read, Write, Delete"
6. Copy the token immediately (you won't see it again)
7. Add it as `DOCKERHUB_TOKEN` in GitHub secrets

## Usage Examples

### Pulling Images

```bash
# Pull latest cognee-mcp
docker pull lvarming/cognee-mcp:latest

# Pull specific version
docker pull lvarming/cognee-mcp:0.4.0

# Pull by commit SHA
docker pull lvarming/cognee-mcp:abc1234

# Pull latest cognee-frontend
docker pull lvarming/cognee-frontend:latest

# Pull specific version
docker pull lvarming/cognee-frontend:1.0.0
```

### Running cognee-mcp

```bash
docker run --rm -it \
  --name cognee-mcp \
  --network bridge \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://cognee-api:8000 \
  -e API_TOKEN=your_token_here \
  -p 8001:8000 \
  lvarming/cognee-mcp:latest
```

### Running cognee-frontend

```bash
docker run --rm -it \
  --name cognee-frontend \
  -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  -e NEXT_PUBLIC_ENABLE_NOTEBOOKS=false \
  lvarming/cognee-frontend:latest
```

## Workflow Features

### Build Optimization
- **Layer caching**: Uses Docker registry cache to speed up builds
- **Multi-stage builds**: Both Dockerfiles use optimized multi-stage builds
- **Dependency caching**: Leverages Docker BuildKit cache mounts

### Security & Best Practices
- **No force push**: Workflows never force push to remote
- **Fork isolation**: Only runs on Varming73/cognee, not on upstream PRs
- **Token-based auth**: Uses secure access tokens, not passwords
- **Immutable tags**: Version and SHA tags are immutable for reproducibility

### Monitoring & Verification
Each workflow includes:
- **Job summaries**: Detailed build information in GitHub Actions UI
- **Image verification**: Pulls the built image to confirm availability
- **Tag listing**: Shows all tags pushed for each build
- **Pull commands**: Provides ready-to-use pull commands

## Triggering Builds

### Automatic Trigger (Recommended)
Simply push changes to relevant directories:

```bash
# Make changes to cognee-mcp
git add cognee-mcp/
git commit -m "feat: update MCP server configuration"
git push origin main

# Make changes to cognee-frontend
git add cognee-frontend/
git commit -m "feat: enhance UI dashboard"
git push origin main
```

### Manual Trigger
1. Go to Actions tab in GitHub
2. Select the workflow (cognee-mcp or cognee-frontend)
3. Click "Run workflow"
4. Optionally specify a custom tag
5. Click "Run workflow" button

### Version Tag Release
```bash
# Update version in pyproject.toml or package.json first
git add cognee-mcp/pyproject.toml
git commit -m "chore: bump version to 0.5.0"
git push origin main

# Create and push version tag
git tag v0.5.0
git push origin v0.5.0
```

## Monitoring Builds

### GitHub Actions UI
1. Navigate to https://github.com/Varming73/cognee/actions
2. Click on the specific workflow run
3. Monitor build progress in real-time
4. View build summary with all published tags

### Docker Hub
1. Visit your repository pages:
   - https://hub.docker.com/r/lvarming/cognee-mcp/tags
   - https://hub.docker.com/r/lvarming/cognee-frontend/tags
2. Verify new tags appear after successful builds
3. Check image sizes and architectures

## Troubleshooting

### Build Fails: Authentication Error
**Problem**: `denied: requested access to the resource is denied`

**Solution**:
1. Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are set correctly
2. Ensure the Docker Hub token has "Read, Write, Delete" permissions
3. Check that the token hasn't expired

### Build Fails: Repository Not Found
**Problem**: `repository does not exist or may require 'docker login'`

**Solution**:
1. Create the Docker Hub repositories manually:
   - Go to https://hub.docker.com
   - Click "Create Repository"
   - Create `cognee-mcp` and `cognee-frontend` repositories
   - Set visibility to Public or Private as desired

### Workflow Doesn't Trigger
**Problem**: Push to main doesn't trigger the workflow

**Solution**:
1. Check that changes were made to watched paths:
   - `cognee-mcp/**` for MCP workflow
   - `cognee-frontend/**` for frontend workflow
2. Verify the workflow file is on the main branch
3. Check workflow file syntax using the Actions tab

### Multi-platform Build Fails
**Problem**: `failed to solve: failed to push`

**Solution**:
1. This is usually temporary - retry the workflow
2. Check Docker Hub status: https://status.docker.com
3. Reduce platforms to `linux/amd64` only if ARM builds consistently fail

## Maintenance

### Updating Workflows
When modifying the workflow files:
1. Test changes on a feature branch first
2. Verify workflow syntax using GitHub's workflow editor
3. Do a test run with manual dispatch before merging to main

### Version Bumping
1. Update version in `pyproject.toml` or `package.json`
2. Commit and push to main
3. Optionally create a git tag for the release
4. New images will be built automatically

### Security Updates
Regularly update workflow dependencies:
- `actions/checkout` - Currently v4
- `docker/setup-qemu-action` - Currently v3
- `docker/setup-buildx-action` - Currently v3
- `docker/login-action` - Currently v3
- `docker/build-push-action` - Currently v5

## Integration with README Badges

Both README files now include Docker Hub badges showing:
- **Build Status**: Green/red badge indicating last build status
- **Docker Version**: Latest version tag available
- **Pull Count**: Total number of image pulls
- **Image Size**: Compressed image size

These badges automatically update when new images are published.

## Best Practices

1. **Version Semantics**: Follow semantic versioning (MAJOR.MINOR.PATCH)
2. **Changelog**: Document changes in commit messages and release notes
3. **Testing**: Test Docker images locally before relying on CI/CD
4. **Monitoring**: Regularly check build status and Docker Hub metrics
5. **Security**: Rotate Docker Hub access tokens periodically
6. **Documentation**: Keep this document updated when workflows change

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [Docker Buildx Documentation](https://docs.docker.com/buildx/working-with-buildx/)
- [Semantic Versioning](https://semver.org/)

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review this documentation
3. Open an issue on the repository
4. Check Docker Hub status page for service issues

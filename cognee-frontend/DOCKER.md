# Docker Setup for Cognee Frontend

This document provides comprehensive information about the Docker setup for the Cognee frontend application.

## Overview

The Cognee frontend uses Next.js 15 with React 19 and includes two Dockerfile configurations:

- **Dockerfile** - Production-optimized multi-stage build
- **Dockerfile.dev** - Development environment with hot-reloading

## Production Build (Dockerfile)

### Features

1. **Multi-stage Build**
   - Stage 1 (deps): Production dependencies only
   - Stage 2 (builder): Full build with all dependencies
   - Stage 3 (runner): Minimal production runtime

2. **Security Best Practices**
   - Non-root user (nextjs:nodejs with UID/GID 1001)
   - Minimal Alpine base image
   - Security updates applied
   - No sensitive files copied (.dockerignore)

3. **Optimization**
   - Layer caching for dependencies
   - Production-only node_modules in final image
   - Clean npm cache after installation
   - Optimized Next.js build output

4. **Production Features**
   - Health check endpoint monitoring
   - Proper signal handling with dumb-init
   - Next.js telemetry disabled
   - Environment variable support

### Building the Production Image

```bash
# From the cognee-frontend directory
docker build -t cognee-frontend:latest .

# Or from the project root
docker build -t cognee-frontend:latest -f cognee-frontend/Dockerfile cognee-frontend/
```

### Running the Production Container

```bash
# Basic run
docker run -p 3000:3000 cognee-frontend:latest

# With environment variables
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_BACKEND_API_URL=http://backend:8000/api \
  cognee-frontend:latest

# With custom port
docker run -p 8080:3000 \
  -e PORT=3000 \
  cognee-frontend:latest
```

## Development Build (Dockerfile.dev)

### Features

1. **Hot Reloading**
   - Next.js development server with fast refresh
   - Volume mounts for live code changes

2. **Debugging Support**
   - Port 9229 exposed for Node.js debugging
   - Development environment variables

### Building the Development Image

```bash
# From the cognee-frontend directory
docker build -t cognee-frontend:dev -f Dockerfile.dev .
```

### Running the Development Container

```bash
# With volume mounts for hot-reloading
docker run -p 3000:3000 -p 9229:9229 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/public:/app/public \
  cognee-frontend:dev
```

## Docker Compose Integration

### Production Profile

```bash
# Start with UI profile (production mode)
docker-compose --profile ui up frontend
```

### Development Override

Create a `docker-compose.override.yml` for development:

```yaml
services:
  frontend:
    build:
      dockerfile: Dockerfile.dev
    volumes:
      - ./cognee-frontend/src:/app/src
      - ./cognee-frontend/public:/app/public
    ports:
      - "3000:3000"
      - "9229:9229"
    environment:
      - NODE_ENV=development
```

Then run:

```bash
docker-compose --profile ui up frontend
```

## Environment Variables

### Build-time Variables

- `NODE_ENV` - Set to 'production' for optimized builds
- `NEXT_TELEMETRY_DISABLED` - Disables Next.js telemetry (set to 1)

### Runtime Variables

- `NEXT_PUBLIC_BACKEND_API_URL` - Backend API URL (required)
- `PORT` - Application port (default: 3000)
- Additional Auth0 variables (see .env.template)

### Using .env Files

```bash
# Create .env file from template
cp .env.template .env

# Edit with your values
nano .env

# Run with env file
docker run -p 3000:3000 --env-file .env cognee-frontend:latest
```

## Health Checks

The production container includes a health check that:

- Runs every 30 seconds
- Has a 30-second timeout
- Allows 5 seconds for startup
- Retries 3 times before marking unhealthy

Check container health:

```bash
docker ps
docker inspect --format='{{.State.Health.Status}}' <container-id>
```

## Image Size Optimization

The multi-stage build significantly reduces the final image size:

- **Before**: ~500MB (with build dependencies)
- **After**: ~150-200MB (production only)

View image size:

```bash
docker images cognee-frontend
```

## Security Considerations

1. **Non-root User**
   - Container runs as user 'nextjs' (UID 1001)
   - Files owned by nextjs:nodejs

2. **Minimal Attack Surface**
   - Alpine base image (minimal packages)
   - Only production dependencies
   - Security updates applied

3. **No Secrets in Image**
   - .dockerignore prevents .env files
   - Use environment variables or secrets management

4. **Signal Handling**
   - dumb-init ensures proper signal forwarding
   - Graceful shutdowns in orchestration

## Troubleshooting

### Build Failures

```bash
# Check build logs
docker build --progress=plain -t cognee-frontend:latest .

# Build without cache
docker build --no-cache -t cognee-frontend:latest .
```

### Runtime Issues

```bash
# View container logs
docker logs <container-id>

# Interactive shell
docker run -it cognee-frontend:latest sh

# Check health
docker inspect <container-id> | grep Health -A 10
```

### Common Issues

1. **Port Already in Use**
   ```bash
   # Use different host port
   docker run -p 3001:3000 cognee-frontend:latest
   ```

2. **Environment Variables Not Working**
   ```bash
   # Verify NEXT_PUBLIC_ prefix for client-side vars
   # Check .env file or -e flags
   ```

3. **Permission Errors**
   ```bash
   # Check file ownership in volumes
   # Ensure nextjs user (1001) has access
   ```

## Performance Tuning

### Build Cache

Leverage Docker layer caching:

```bash
# Dependencies change infrequently - cached
COPY package.json package-lock.json ./
RUN npm ci

# Source code changes frequently - not cached
COPY src ./src
```

### Resource Limits

In docker-compose.yml:

```yaml
services:
  frontend:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

### Node.js Memory

For large builds:

```bash
docker build --build-arg NODE_OPTIONS="--max-old-space-size=4096" .
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Build Docker Image
  run: |
    docker build -t cognee-frontend:${{ github.sha }} .

- name: Push to Registry
  run: |
    docker tag cognee-frontend:${{ github.sha }} registry/cognee-frontend:latest
    docker push registry/cognee-frontend:latest
```

### GitLab CI Example

```yaml
build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## Best Practices

1. **Use Specific Base Image Tags**
   - Current: `node:22-alpine`
   - Avoid: `node:latest` (unpredictable)

2. **Multi-stage Builds**
   - Separate build and runtime stages
   - Only copy necessary artifacts

3. **Layer Ordering**
   - Least frequently changed first
   - Most frequently changed last

4. **Security Scanning**
   ```bash
   docker scan cognee-frontend:latest
   ```

5. **Regular Updates**
   ```bash
   # Update base image
   docker pull node:22-alpine
   docker build --no-cache -t cognee-frontend:latest .
   ```

## References

- [Next.js Docker Documentation](https://nextjs.org/docs/deployment)
- [Docker Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Alpine Linux Package Management](https://wiki.alpinelinux.org/wiki/Alpine_Package_Keeper)
- [Node.js Docker Best Practices](https://github.com/nodejs/docker-node/blob/main/docs/BestPractices.md)

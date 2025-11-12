# Docker Quick Reference - Cognee Frontend

## Quick Commands

### Build

```bash
# Production build
docker build -t cognee-frontend:latest .

# Development build
docker build -f Dockerfile.dev -t cognee-frontend:dev .

# Build without cache
docker build --no-cache -t cognee-frontend:latest .

# Build with verbose output
docker build --progress=plain -t cognee-frontend:latest .
```

### Run

```bash
# Production (basic)
docker run -p 3000:3000 cognee-frontend:latest

# Production (with env vars)
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_BACKEND_API_URL=http://backend:8000/api \
  cognee-frontend:latest

# Production (detached with name)
docker run -d --name frontend -p 3000:3000 cognee-frontend:latest

# Production (with .env file)
docker run -p 3000:3000 --env-file .env cognee-frontend:latest

# Development (with hot-reload)
docker run -p 3000:3000 -p 9229:9229 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/public:/app/public \
  cognee-frontend:dev
```

### Docker Compose

```bash
# Start with UI profile
docker-compose --profile ui up frontend

# Start in background
docker-compose --profile ui up -d frontend

# Rebuild and start
docker-compose --profile ui up --build frontend

# Stop
docker-compose --profile ui down

# View logs
docker-compose --profile ui logs -f frontend
```

### Debug

```bash
# View logs
docker logs <container-id>

# Follow logs
docker logs -f <container-id>

# Interactive shell
docker run -it cognee-frontend:latest sh

# Execute command in running container
docker exec -it <container-id> sh

# Inspect container
docker inspect <container-id>

# Check health status
docker inspect --format='{{.State.Health.Status}}' <container-id>

# View resource usage
docker stats <container-id>
```

### Cleanup

```bash
# Stop container
docker stop <container-id>

# Remove container
docker rm <container-id>

# Remove image
docker rmi cognee-frontend:latest

# Stop and remove
docker stop <container-id> && docker rm <container-id>

# Prune unused images
docker image prune

# Prune everything
docker system prune -a
```

## Environment Variables

### Build-time
- `NODE_ENV=production` - Set production mode
- `NEXT_TELEMETRY_DISABLED=1` - Disable telemetry

### Runtime
- `NEXT_PUBLIC_BACKEND_API_URL` - Backend API URL (required)
- `PORT` - Application port (default: 3000)

## Port Reference

- `3000` - Next.js application
- `9229` - Node.js debugging (dev only)

## Common Workflows

### Local Development
```bash
# 1. Build dev image
docker build -f Dockerfile.dev -t cognee-frontend:dev .

# 2. Run with hot-reload
docker run -p 3000:3000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/public:/app/public \
  cognee-frontend:dev

# 3. Access at http://localhost:3000
```

### Production Testing
```bash
# 1. Build production image
docker build -t cognee-frontend:latest .

# 2. Run production container
docker run -d --name test-frontend -p 3000:3000 cognee-frontend:latest

# 3. Test
curl http://localhost:3000

# 4. Cleanup
docker stop test-frontend && docker rm test-frontend
```

### CI/CD Pipeline
```bash
# 1. Build with tag
docker build -t registry/cognee-frontend:${VERSION} .

# 2. Run tests
docker run --rm registry/cognee-frontend:${VERSION} npm test

# 3. Security scan
docker scan registry/cognee-frontend:${VERSION}

# 4. Push to registry
docker push registry/cognee-frontend:${VERSION}
docker tag registry/cognee-frontend:${VERSION} registry/cognee-frontend:latest
docker push registry/cognee-frontend:latest
```

## Troubleshooting

### Port Already in Use
```bash
# Use different port
docker run -p 3001:3000 cognee-frontend:latest
```

### Permission Errors
```bash
# Check file ownership
ls -la src/

# Fix ownership (UID 1001 for container user)
sudo chown -R 1001:1001 src/
```

### Build Failures
```bash
# Clear cache
docker builder prune

# Rebuild without cache
docker build --no-cache -t cognee-frontend:latest .
```

### Container Won't Start
```bash
# Check logs
docker logs <container-id>

# Verify environment variables
docker inspect <container-id> | grep Env

# Test interactively
docker run -it cognee-frontend:latest sh
```

## Health Check

```bash
# View health status
docker ps

# Detailed health info
docker inspect <container-id> | grep Health -A 20

# Manual health check
curl http://localhost:3000/
```

## Image Management

```bash
# List images
docker images cognee-frontend

# Check image size
docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" cognee-frontend

# Image history (layer sizes)
docker history cognee-frontend:latest

# Remove old images
docker images -f "dangling=true" -q | xargs docker rmi
```

## Security

```bash
# Verify non-root user
docker run --rm cognee-frontend:latest id
# Expected: uid=1001(nextjs) gid=1001(nodejs)

# Security scan
docker scan cognee-frontend:latest

# Check for vulnerabilities
trivy image cognee-frontend:latest
```

## Performance

```bash
# View real-time stats
docker stats <container-id>

# Resource limits
docker run -p 3000:3000 \
  --cpus="1.0" \
  --memory="512m" \
  cognee-frontend:latest

# Build with BuildKit (faster)
DOCKER_BUILDKIT=1 docker build -t cognee-frontend:latest .
```

## Registry Operations

```bash
# Login to registry
docker login registry.example.com

# Tag for registry
docker tag cognee-frontend:latest registry.example.com/cognee-frontend:latest

# Push to registry
docker push registry.example.com/cognee-frontend:latest

# Pull from registry
docker pull registry.example.com/cognee-frontend:latest
```

## File Locations

```
/Users/lvarming/it-setup/projects/cognee_og/cognee-frontend/
├── Dockerfile                    # Production build
├── Dockerfile.dev               # Development build
├── .dockerignore                # Build exclusions
├── docker-compose.frontend.yml  # Standalone compose
├── DOCKER.md                    # Full documentation
├── DOCKER_OPTIMIZATION_SUMMARY.md  # Detailed summary
└── DOCKER_QUICK_REFERENCE.md   # This file
```

## Support

For detailed information, see:
- DOCKER.md - Comprehensive documentation
- DOCKER_OPTIMIZATION_SUMMARY.md - Optimization details
- README.md - Project overview

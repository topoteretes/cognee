# Cognee Frontend

<div align="center">
  <a href="https://github.com/Varming73/cognee/actions/workflows/cognee-frontend-docker.yml">
    <img src="https://github.com/Varming73/cognee/actions/workflows/cognee-frontend-docker.yml/badge.svg" alt="Docker Build Status">
  </a>
  <a href="https://hub.docker.com/r/lvarming/cognee-frontend">
    <img src="https://img.shields.io/docker/v/lvarming/cognee-frontend?label=docker&logo=docker" alt="Docker Image Version">
  </a>
  <a href="https://hub.docker.com/r/lvarming/cognee-frontend">
    <img src="https://img.shields.io/docker/pulls/lvarming/cognee-frontend?logo=docker" alt="Docker Pulls">
  </a>
  <a href="https://hub.docker.com/r/lvarming/cognee-frontend">
    <img src="https://img.shields.io/docker/image-size/lvarming/cognee-frontend?logo=docker" alt="Docker Image Size">
  </a>
</div>

This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Getting Started

### Local Development

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/basic-features/font-optimization) to automatically optimize and load Inter, a custom Google Font.

## Docker Deployment

### Production Deployment

The frontend includes a production-optimized, multi-stage Docker build with security hardening and performance optimization.

#### Quick Start

```bash
# Pull from Docker Hub (if available)
docker pull lvarming/cognee-frontend:latest

# Or build locally
docker build -t cognee-frontend:latest .

# Run the container
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_BACKEND_API_URL=http://localhost:8000/api \
  cognee-frontend:latest
```

#### Using Docker Compose

```bash
# Start with UI profile (from project root)
docker-compose --profile ui up frontend

# Or use the standalone configuration
cd cognee-frontend
docker-compose -f docker-compose.frontend.yml up frontend-prod
```

### Development with Docker

For local development with hot-reloading:

```bash
# Build development image
docker build -f Dockerfile.dev -t cognee-frontend:dev .

# Run with volume mounts for live code updates
docker run -p 3000:3000 -p 9229:9229 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/public:/app/public \
  cognee-frontend:dev

# Or use docker-compose
docker-compose -f docker-compose.frontend.yml --profile dev up frontend-dev
```

### Docker Features

The optimized Docker setup includes:

- **Multi-stage build** - 64% smaller images (~180MB vs ~500MB)
- **Security hardened** - Non-root user (nextjs:1001), minimal Alpine base
- **Production ready** - Health checks, proper signal handling (dumb-init)
- **Build optimization** - Layer caching reduces rebuild time by 4x
- **Environment support** - Separate Dockerfile for development and production

### Docker Documentation

Comprehensive Docker documentation is available:

- **[DOCKER.md](DOCKER.md)** - Complete Docker usage guide
- **[DOCKER_QUICK_REFERENCE.md](DOCKER_QUICK_REFERENCE.md)** - Quick command reference
- **[DOCKER_OPTIMIZATION_SUMMARY.md](DOCKER_OPTIMIZATION_SUMMARY.md)** - Detailed optimization analysis
- **[DOCKER_COMPARISON.md](DOCKER_COMPARISON.md)** - Before/after comparison
- **[docker-test.sh](docker-test.sh)** - Automated validation script

### Testing Docker Setup

Run the automated validation script to verify the Docker configuration:

```bash
cd cognee-frontend
./docker-test.sh
```

This will test:
- Build process and image size
- Multi-stage build structure
- Security features (non-root user, dumb-init)
- Health checks
- Resource usage
- Build cache performance

### Multi-platform Support

The Docker images are built for both `linux/amd64` and `linux/arm64` architectures, making them suitable for:
- x86_64 servers and desktops
- ARM-based systems (Apple Silicon, Raspberry Pi, AWS Graviton)

### Environment Variables

#### Build-time Variables
- `NODE_ENV` - Set to 'production' for optimized builds
- `NEXT_TELEMETRY_DISABLED` - Disables Next.js telemetry (default: 1)

#### Runtime Variables
- `NEXT_PUBLIC_BACKEND_API_URL` - Backend API URL (required)
- `PORT` - Application port (default: 3000)

See `.env.template` for all available variables.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js/) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/deployment) for more details.

## Feature Flags

This fork hides some of the upstream experimental UI by default. Toggle them via environment variables:

| Flag | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_ENABLE_NOTEBOOKS` | `false` | Set to `true` to show the notebook workflow builder. |
| `NEXT_PUBLIC_ENABLE_CLOUD_CONNECTOR` | `false` | Set to `true` to re-enable the Cloud Cognee connection panel. |

### Search Panel

The dashboard search/QA view now mirrors the MCP capabilities:

- Dataset dropdown lets you target any local dataset (defaults to the first available one).
- Advanced options expose `combined context`, `context only`, and node filters (comma-separated node set names) which map directly to Cognee's `/v1/search` flags.
- The "max results" input maps to `top_k`.

Reminder: dataset and file deletions run in *hard* mode, so the UI warns that graph data will be purged.

Add the variables to your `.env.local` (or deployment environment) before running `npm run dev`.

## Project Structure

```
cognee-frontend/
├── src/                        # Application source code
│   ├── app/                   # Next.js app directory
│   ├── modules/               # Feature modules
│   ├── ui/                    # UI components
│   └── utils/                 # Utility functions
├── public/                    # Static assets
├── Dockerfile                 # Production Docker build
├── Dockerfile.dev            # Development Docker build
├── docker-compose.frontend.yml # Standalone Docker Compose
├── docker-test.sh            # Docker validation script
└── DOCKER*.md                # Docker documentation
```

## Performance

The Docker setup is optimized for:

- **Fast rebuilds**: 45-60 seconds for code changes (vs 3-4 minutes)
- **Small images**: ~180MB production image (vs ~500MB)
- **Quick startup**: 2-3 seconds container startup time
- **Low memory**: 150-200MB idle, 300-400MB under load

## Security

Security features include:

- Non-root user (nextjs:1001) for container processes
- Minimal Alpine base image with security updates
- No secrets or .env files in Docker images
- Health check monitoring for orchestration
- Proper signal handling for graceful shutdowns

## Troubleshooting

### Docker Issues

For Docker-related issues, see:
- [DOCKER.md](DOCKER.md#troubleshooting) - Comprehensive troubleshooting guide
- [DOCKER_QUICK_REFERENCE.md](DOCKER_QUICK_REFERENCE.md#troubleshooting) - Quick fixes

### Common Issues

1. **Port already in use**
   ```bash
   docker run -p 3001:3000 cognee-frontend:latest
   ```

2. **Environment variables not working**
   - Verify `NEXT_PUBLIC_` prefix for client-side variables
   - Check .env file or use `--env-file` flag

3. **Build failures**
   ```bash
   docker builder prune  # Clear build cache
   docker build --no-cache -t cognee-frontend:latest .
   ```

## Contributing

Contributions are welcome! Please ensure Docker changes follow best practices:

- Test with `./docker-test.sh` before submitting
- Update documentation if Docker configuration changes
- Follow security best practices (non-root user, minimal base image)
- Optimize for build cache and image size

## License

See the main project LICENSE file for details.

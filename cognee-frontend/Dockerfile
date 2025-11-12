# ==============================================
# Multi-stage Dockerfile for Next.js Frontend
# Optimized for production with security best practices
# ==============================================

# ============== Stage 1: Dependencies ==============
# Install dependencies in a separate stage for better caching
FROM node:22-alpine AS deps

# Install security updates
RUN apk update && apk upgrade && apk add --no-cache libc6-compat

WORKDIR /app

# Copy package files for dependency installation
COPY package.json package-lock.json ./

# Install production dependencies only
# Use --omit=dev for production dependencies
# Keep cache clean to reduce image size
RUN npm ci --omit=dev && \
    npm cache clean --force

# ============== Stage 2: Builder ==============
# Build the Next.js application
FROM node:22-alpine AS builder

WORKDIR /app

# Copy installed dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy package files
COPY package.json package-lock.json ./

# Install all dependencies (including devDependencies for build)
RUN npm ci && npm cache clean --force

# Copy source code and configuration files
COPY src ./src
COPY public ./public
COPY next.config.mjs .
COPY postcss.config.mjs .
COPY tsconfig.json .
COPY tailwindcss.config.js* ./
COPY .env.template .

# Set environment variables for build
# Disable Next.js telemetry
ENV NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production

# Build the Next.js application
# This creates the optimized production build in .next folder
RUN npm run build

# ============== Stage 3: Runner (Production) ==============
# Final stage for running the application
FROM node:22-alpine AS runner

# Install security updates and dumb-init for proper signal handling
RUN apk update && apk upgrade && apk add --no-cache dumb-init

WORKDIR /app

# Set production environment
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

# Create a non-root user for security
# Using fixed UID/GID for consistency across deployments
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy necessary files from builder
# Copy public folder (static assets)
COPY --from=builder /app/public ./public

# Copy Next.js build output
# Change ownership to non-root user
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next

# Copy node_modules from deps stage (production only)
COPY --from=deps --chown=nextjs:nodejs /app/node_modules ./node_modules

# Copy package.json for running the app
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json

# Copy Next.js configuration
COPY --from=builder --chown=nextjs:nodejs /app/next.config.mjs ./next.config.mjs

# Switch to non-root user
USER nextjs

# Expose the application port
EXPOSE 3000

# Health check to ensure the application is running
# Checks the Next.js health endpoint every 30 seconds
# Allows 3 retries with 30 second timeout
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/', (r) => {if (r.statusCode !== 200) throw new Error(r.statusCode)})"

# Use dumb-init to handle signals properly
# This ensures graceful shutdowns in container orchestration
ENTRYPOINT ["dumb-init", "--"]

# Start the Next.js production server
CMD ["npm", "start"]

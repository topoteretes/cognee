# syntax=docker/dockerfile:1

# Production image for the cognee UI.
#
# The backend it talks to is chosen at *run* time via COGNEE_BACKEND_URL, not
# at build time, so one published image works against a local Docker backend,
# a `cognee-cli` backend, or a remote one. See
# cognee-frontend/src/modules/config/runtimeConfig.ts for how that is wired.

# ---------------------------------------------------------------- deps ------
FROM node:22-alpine AS deps
WORKDIR /app

# Next's native toolchain (@tailwindcss/oxide, lightningcss) expects glibc
# symbols that musl does not provide on its own.
RUN apk add --no-cache libc6-compat

COPY package.json package-lock.json ./
# `npm ci` fails on a lockfile that drifted from package.json, which is the
# class of breakage that reached main as COG-6035.
RUN npm ci

# ------------------------------------------------------------- builder ------
FROM node:22-alpine AS builder
WORKDIR /app

RUN apk add --no-cache libc6-compat

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# This one genuinely belongs at build time: the cloud shell is dead code in the
# open-source image, so there is nothing to configure per container.
ENV NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT=false
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# ----------------------------------------------------------------- dev ------
# Hot-reloading server for contributors: `docker compose --profile ui-dev up`.
# Never published. `runner` is the last stage, so it stays the default target
# and BuildKit skips this one entirely unless it is asked for by name.
FROM node:22-alpine AS dev
WORKDIR /app

RUN apk add --no-cache libc6-compat

COPY --from=deps /app/node_modules ./node_modules
COPY . .

ENV NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT=false
ENV NEXT_TELEMETRY_DISABLED=1

EXPOSE 3000
CMD ["npm", "run", "dev"]

# -------------------------------------------------------------- runner ------
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Empty means "ask the browser": the UI then derives the backend host from the
# address it was loaded from, which is right for the common localhost setup.
# Set it to point somewhere else, e.g.
# -e COGNEE_BACKEND_URL=https://cognee.example.com. The browser calls the
# backend directly, so this is the address as seen from the browser, never a
# compose service name.
ENV COGNEE_BACKEND_URL=""

RUN addgroup -g 1001 -S nodejs \
 && adduser -u 1001 -S nextjs -G nodejs

# `output: "standalone"` emits a server plus only the node_modules the app
# actually reaches: no dev dependencies, no npm install in this stage.
RUN mkdir .next && chown nextjs:nodejs .next

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

USER nextjs
EXPOSE 3000

# Probes the runtime-config route rather than "/": it is the cheapest endpoint
# that proves both that the server is up and that COGNEE_BACKEND_URL resolved.
# node:22 has a global fetch, so the runtime image needs no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||3000)+'/api/runtime-config').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["node", "server.js"]

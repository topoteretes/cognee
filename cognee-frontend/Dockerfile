# Build stage
FROM node:22-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci
RUN npm rebuild lightningcss

COPY src ./src
COPY public ./public
COPY types ./types
COPY next.config.mjs .
COPY next.config.ts .
COPY postcss.config.mjs .
COPY eslint.config.mjs .
COPY tsconfig.json .

ENV NEXT_PUBLIC_LOCAL_API_URL=__NEXT_PUBLIC_LOCAL_API_URL__
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npm run build

# Production stage
FROM node:22-alpine
WORKDIR /app

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["/bin/sh", "-c", "find .next -name '*.js' | xargs sed -i \"s|__NEXT_PUBLIC_LOCAL_API_URL__|$NEXT_PUBLIC_LOCAL_API_URL|g\" && node server.js"]

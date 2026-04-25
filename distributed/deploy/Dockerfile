# Pre-built image for distributed/deploy/daytona_onboarding_demo.py
#
# Bakes the slow parts of each sandbox boot — Node, Claude Code CLI, cognee,
# the Moss vector adapter, and the daytona Python SDK — so each sandbox can
# start in ~30s instead of ~5-7 min. Volume-based cross-agent state sharing
# is unaffected; sandboxes built from this image are still fully isolated.
#
# Build (pin to a specific cognee commit so the image is deterministic):
#
#   docker build \
#     --platform linux/amd64 \
#     --build-arg COGNEE_REF=graphskills-on-agentic \
#     -t cognee-skills-hackathon:latest \
#     -f distributed/deploy/Dockerfile .
#
# Push (anywhere participants can pull from — GHCR public works well):
#
#   docker tag cognee-skills-hackathon:latest \
#       ghcr.io/<org>/cognee-skills-hackathon:latest
#   docker push ghcr.io/<org>/cognee-skills-hackathon:latest
#
# Use it from the demo by switching the `Image.base(...)` call in
# `deploy_claude_code_agent` from "python:3.12-slim-trixie" to the image
# tag above, and removing the in-script install commands (Node, Claude
# Code, cognee, moss adapter, daytona pkg) — they're now baked in.

FROM python:3.12-slim-trixie

ARG COGNEE_REF=graphskills-on-agentic

# System deps for native builds + curl (Node tarball + cognee VCS install)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git xz-utils build-essential && \
    rm -rf /var/lib/apt/lists/*

# Node.js 22 via prebuilt tarball (matches the runtime install path the demo
# script uses today — more reliable than NodeSource's apt repo, which has
# intermittent connection-reset issues).
ARG NODE_VERSION=v22.16.0
RUN curl -fsSL "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz" \
        | tar -xJ -C /usr/local --strip-components=1 && \
    node --version && npm --version

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Python packages: cognee (pinned), Moss vector adapter, daytona SDK.
# Native deps inside cognee (kuzu, lancedb, sentence-transformers, etc.)
# get downloaded as wheels here once and stay in the image.
RUN pip install --no-cache-dir \
        "git+https://github.com/topoteretes/cognee.git@${COGNEE_REF}" \
        cognee-community-vector-adapter-moss==0.1.1 \
        daytona

# Pre-create the directories the demo writes to. Do NOT create the `agent`
# user here — the demo script creates it at runtime via `useradd`, and a
# duplicate create would error. The script also chowns these dirs after
# the user exists.
RUN mkdir -p /workspace /var/cognee-state /shared-cognee /opt/cognee-integrations \
             /opt/cognee-demo-skills

WORKDIR /workspace

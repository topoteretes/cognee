# Docker Quick Reference

## Start full stack (without frontend)

```bash
docker compose --profile postgres --profile neo4j --profile redis up -d postgres neo4j redis cognee
```

## Start full stack (with MCP)

```bash
docker compose --profile postgres --profile neo4j --profile redis --profile mcp up -d postgres neo4j redis cognee cognee-mcp
```

## Check status

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## Restart only API

```bash
docker compose restart cognee
```

## Stop all

```bash
docker compose down
```

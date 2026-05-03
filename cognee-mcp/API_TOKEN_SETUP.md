# API Token Setup for Cognee

## Overview

The Cognee API uses JWT tokens for authentication. Tokens must have:
- `sub`: User UUID (the user's ID in the database)
- `aud`: Must be `['fastapi-users:auth']`
- `exp`: Expiration timestamp

## Method 1: Login via API (Recommended)

### Step 1: Login to get token

```bash
curl -X POST "https://your-domain.com/cognee/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your@email.com&password=your-password"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Step 2: Use the token

```bash
curl -X POST "https://your-domain.com/cognee/api/v1/remember" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -F "data=@file.txt" \
  -F "datasetName=main_dataset"
```

## Method 2: Generate JWT manually

### Prerequisites

You need:
1. `FASTAPI_USERS_JWT_SECRET` - from your `.env` file
2. User UUID - from your user account in the database

### Generate Token

```bash
python3 -c "
from datetime import datetime, timedelta
import jwt

secret = 'YOUR_FASTAPI_USERS_JWT_SECRET'
payload = {
    'sub': 'YOUR_USER_UUID',
    'aud': ['fastapi-users:auth'],
    'exp': datetime.utcnow() + timedelta(days=365)
}
token = jwt.encode(payload, secret, algorithm='HS256')
print(token)
"
```

### Use the Token

Same as Method 1, Step 2.

## Method 3: Register + Login

### Step 1: Enable Registration (if disabled)

Edit `cognee/api/client.py`:
```python
from cognee.api.v1.users.routers import (
    get_auth_router,
    get_register_router,  # Enable this
    ...
)

# Also uncomment the router registration:
app.include_router(
    get_register_router(),
    prefix="/api/v1/auth",
    tags=["auth"],
)
```

### Step 2: Register a new user

```bash
curl -X POST "https://your-domain.com/cognee/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "password": "your-password",
    "is_active": true,
    "is_superuser": false,
    "is_verified": false
  }'
```

### Step 3: Login to get token

Then use Method 1 to login and get your token.

## MCP Server Usage

When using the MCP server with API mode:

```bash
# Using --api-token flag
uv run python -m cognee_mcp.src.server \
  --transport sse \
  --api-url http://localhost:8000 \
  --api-token "your-jwt-token-here"
```

Or set in environment:
```bash
export API_TOKEN="your-jwt-token-here"
```

## Troubleshooting

### 401 Unauthorized

**Common causes:**
1. **Wrong user UUID** - The `sub` claim must match an existing user
2. **Token expired** - Check the `exp` claim
3. **Missing audience** - Must include `aud: ['fastapi-users:auth']`
4. **Secret mismatch** - The secret must match `FASTAPI_USERS_JWT_SECRET`

### Debugging

To test your token locally:

```python
from fastapi_users.jwt import decode_jwt

secret = 'YOUR_SECRET'
token = 'YOUR_TOKEN'

payload = decode_jwt(token, secret, ['fastapi-users:auth'], algorithms=['HS256'])
print(payload)
# Should show: {'sub': '...', 'aud': [...], 'exp': ...}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FASTAPI_USERS_JWT_SECRET` | JWT signing secret |
| `JWT_LIFETIME_SECONDS` | Token lifetime (default: 3600) |
| `REQUIRE_AUTHENTICATION` | Enable auth (default: true) |
| `ENABLE_BACKEND_ACCESS_CONTROL` | Enable access control |
| `API_TOKEN` | Pre-generated token for MCP |

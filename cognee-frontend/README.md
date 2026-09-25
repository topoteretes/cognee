This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Testing Google Drive and Gmail against the SDK

The `/integrations` page has SDK-native Google account cards. These call
`/api/v1/integrations/{google_drive|gmail}` directly with browser credentials;
they do not require Cognee Cloud or its control plane. OAuth secrets belong only
on the Python server, never in `NEXT_PUBLIC_*` variables.

From the repository root, install the SDK extras:

```bash
uv sync --extra google-drive --extra gmail --dev
```

Configure a Google OAuth **Web application** client with the Drive and Gmail APIs
enabled. For local testing, register both exact callback URLs shown below and
allow your test Google account in the OAuth consent configuration. Add these
settings to the repository-root `.env` (do not commit real credentials):

```dotenv
GOOGLE_DRIVE_CLIENT_ID=<Google OAuth client ID>
GOOGLE_DRIVE_CLIENT_SECRET=<Google OAuth client secret>
GOOGLE_DRIVE_REDIRECT_URI=http://localhost:8000/api/v1/integrations/google_drive/callback
GOOGLE_DRIVE_FRONTEND_BASE_URL=http://localhost:3000
GOOGLE_DRIVE_STATE_SECRET=<random secret>
GOOGLE_GMAIL_CLIENT_ID=<Google OAuth client ID>
GOOGLE_GMAIL_CLIENT_SECRET=<Google OAuth client secret>
GOOGLE_GMAIL_REDIRECT_URI=http://localhost:8000/api/v1/integrations/gmail/callback
GOOGLE_GMAIL_FRONTEND_BASE_URL=http://localhost:3000
GOOGLE_GMAIL_STATE_SECRET=<another random secret>
INTEGRATION_CREDENTIALS_KEY=<base64-encoded 32-byte encryption key>
```

Generate each state secret with `openssl rand -hex 32` and the encryption key
with `openssl rand -base64 32`. Keep the encryption key stable across restarts,
or stored Google tokens cannot be decrypted. Configure the usual LLM/embedding
provider settings too: sync runs ingestion and cognification.

Start a **loopback-only, single-user test server with isolated storage** from the
repository root. Reuse the same test root when restarting to retain connections:

```bash
GOOGLE_TEST_ROOT=$(mktemp -d /tmp/cognee-google-ui.XXXXXX)
DB_PATH="$GOOGLE_TEST_ROOT/databases" \
SYSTEM_ROOT_DIRECTORY="$GOOGLE_TEST_ROOT/system" \
DATA_ROOT_DIRECTORY="$GOOGLE_TEST_ROOT/data" \
CACHE_ROOT_DIRECTORY="$GOOGLE_TEST_ROOT/cache" \
COGNEE_LOGS_DIR="$GOOGLE_TEST_ROOT/logs" \
DLT_DATA_DIR="$GOOGLE_TEST_ROOT/dlt" \
PIPELINES_DIR="$GOOGLE_TEST_ROOT/dlt/pipelines" \
DB_PROVIDER=sqlite GRAPH_DATABASE_PROVIDER=ladybug VECTOR_DB_PROVIDER=lancedb \
ENABLE_BACKEND_ACCESS_CONTROL=true REQUIRE_AUTHENTICATION=true \
DEFAULT_USER_PASSWORD='<choose a local test password>' \
AUTH_TOKEN_COOKIE_NAME=cognee_google_test_auth \
CORS_ALLOWED_ORIGINS=http://localhost:3000 \
uv run --no-sync python -m uvicorn cognee.api.client:app --host 127.0.0.1 --port 8000
```

In another terminal, from `cognee-frontend`:

```bash
npm ci
NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT=false COGNEE_BACKEND_URL=http://localhost:8000 \
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open **http://localhost:3000/integrations**. Use `localhost` consistently, not
`127.0.0.1` in the browser: OAuth cookies and redirect origins must match.
The UI still requires a login session for `/users/me`. Sign in as
`default_user@example.com` with the test password you chose. The separate cookie
name avoids replacing another local Cognee instance's login. Authentication is
also required on SDK endpoints. Do not expose this single-user test server publicly.

Keep dataset access control enabled for searches scoped to individual brains.
An existing instance created with it disabled has a shared graph; switching the
flag requires rebuilding its per-dataset indexes, not just restarting the API.

Manual checklist:

1. Connect a test Google account. New accounts start with no resources selected;
   reconnecting retains an explicitly saved scope and does not start an import. Use Refresh to sync.
2. Select a small Drive folder or Gmail test label, then **Save selection** and
   **Refresh**. Multiple Gmail labels are an intersection, not a union.
3. Wait for the last-sync timestamp/counts to change, then inspect the imported
   dataset and query its content. Accepted means queued, not completed; status
   refreshes every 10 seconds. There is no periodic sync scheduler.
   Brains and brain details show persisted ready/remaining counts and per-item
   completion, refreshed every 5 seconds even after closing the connection dialog.
   These totals cover imported items, not files still being downloaded from Google.
4. Modify/delete a test file or message and sync again. Verify the replacement
   and deletion in Cognee, not just the Google connection indicator.
5. Test disconnect retaining data. Reconnect, then separately test the explicit
   delete-data checkbox using only disposable imported data.

An empty selection pauses future ingestion; narrowing a selection does not erase
previously imported data. Missing OAuth settings appear as a server-configuration
error in the connect dialog. After editing backend settings, restart the backend.

Drive and Gmail use incremental remote cursors. A no-change sync still reconciles
that resource's staged documents into Cognee, so a missing local import or failed
processing run can recover without editing the remote file. Unchanged, successfully
processed items retain their completion state.

For cloud deployment, register HTTPS callback/frontend URLs and supply OAuth
credentials, state secrets, and the stable encryption key through server-side
secret management. The current sync guard and background tasks run in one API
process; multiple workers or scheduled sync require a durable queue and distributed
locking. This UI change does not install a scheduler or deploy cloud infrastructure.

## Getting Started

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

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js/) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/deployment) for more details.

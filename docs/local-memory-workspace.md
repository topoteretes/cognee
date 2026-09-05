# Local memory workspace

The SDK and UI on this branch can serve several plugins and people from one
Cognee backend. Each request still uses a particular account and dataset.
Point every client at the same backend URL; sharing a server does not grant
access to every dataset.

## Run the matching SDK and UI

Use this checkout for both services. A published UI image or `cognee-cli -ui`
may contain a different UI revision. Back up existing storage before running
an unreleased SDK against it. Never start two backend processes on the same
embedded database directory.

```sh
uv sync --dev --locked
# Export your existing storage, authentication and model configuration first.
# Keep REQUIRE_AUTHENTICATION=true and ENABLE_BACKEND_ACCESS_CONTROL=true.
uv run uvicorn cognee.api.client:app --host 127.0.0.1 --port 8011
```

In another terminal:

```sh
cd cognee-frontend
npm ci
npm run build
COGNEE_BACKEND_URL=http://localhost:8011 npm start -- --hostname 127.0.0.1 --port 3000
```

Allow the UI origin in the API's `CORS_ALLOWED_ORIGINS`. Set
`SYSTEM_ROOT_DIRECTORY`, `DATA_ROOT_DIRECTORY`, and `CACHE_ROOT_DIRECTORY` to
persistent locations outside the Python environment. Leave `CACHING` enabled.
For a continuously running central server, leave `COGNEE_AGENT_MODE=false`.
The Codex and Claude plugins connect to a healthy existing server without
starting another one. A separately managed server needs its own restart policy.

Open `/workspace` and sign in with your own account. Existing plugin memory
belongs to the account that originally saved it. Creating another UI account
does not transfer that memory. Never put an owner API key in a `NEXT_PUBLIC_`
environment variable or share the owner's login with teammates.

For access from another machine, use a reachable HTTPS backend URL and UI URL,
configure CORS and authentication cookies for that deployment, and protect the
server with your chosen private network or reverse proxy. A browser on another
machine cannot use the server's `localhost` address. Deployment and external
network exposure are separate from the local setup.

## Configure the existing Slack and GitHub integrations

The Sources tab uses the SDK's existing integration endpoints. No Cognee Cloud
account is required. It displays missing setting names without exposing values.
Configure your own provider apps on the **backend**, then restart it:

- Slack: `SLACK_CLIENT_ID`, `SLACK_CLIENT_SECRET`, `SLACK_SIGNING_SECRET`,
  `SLACK_REDIRECT_URI`, `SLACK_FRONTEND_BASE_URL`.
- GitHub App: `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `GITHUB_APP_ID`,
  `GITHUB_APP_SLUG`, `GITHUB_APP_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET`,
  `GITHUB_FRONTEND_BASE_URL`.
- Token encryption: `INTEGRATION_CREDENTIALS_KEYS` (JSON object of key IDs to
  base64-encoded 32-byte keys) and `INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID`.
  The existing single-key `INTEGRATION_CREDENTIALS_KEY` remains supported.
  Preserve encryption keys in backups so existing provider tokens remain usable.

Both frontend base URLs point to the UI origin. OAuth callback URLs are
`<backend>/api/v1/integrations/slack/callback` and
`<backend>/api/v1/integrations/github/callback`. Generic provider event URLs are
`<backend>/api/v1/integrations/<provider>/events`. Slack also uses its existing
command/interaction routes; configure the app against the SDK's Slack router.
Providers must be able to reach webhook endpoints: a purely loopback server
cannot receive their events. Configure a reachable HTTPS endpoint when needed.

Select repositories in your GitHub App installation. The existing connector
indexes repository code; it does not promise an issues/PR discussion archive.
The Slack app saves selected messages and notes. Its channel allowlist controls
where slash commands run; it is not a channel-history importer. Empty channel
selection means unrestricted commands in the existing SDK, so the UI requires
at least one channel when restriction is enabled.

Connections belong to the authenticated SDK account. Dataset permissions
control access to ingested memory separately from provider installation access.

## Agent permissions and team access

The Agent access tab shows direct and inherited dataset permissions. Change
Read, Write, Share or Delete for a person, agent, role or team. Each toggle
changes only that permission and reloads the saved grants. Revoking direct
Read cannot remove Read inherited from a team or role.

Plugin identities are created with `create_only=true`, so an existing identity
returns a conflict rather than rotating its key. Copy a newly created key once
and use the plugin's Manage Access reconnect workflow. Existing sessions need
a fresh host session to use a changed identity. Dataset access alone does not
change the plugin's selected write dataset or graph-read selection; use the
plugin's explicit dataset/read-selection workflow for those bindings.

The Team tab creates invitations for a specific email address. Copy the code
and deliver it yourself. The recipient creates an account on this same server,
accepts the code, then selects the team. Codes expire after seven days, are
stored hashed, can be revoked, and are accepted once. No email is sent. Only the
team owner creates invitations. Joining makes existing team grants effective;
it does not grant private datasets. Grant Read and Write to let a colleague
continue work; grant Share if they should also manage access. This is shared
access to memory, not a transfer of the original account or provider tokens.

## Review and promote memory

1. Choose the source dataset and a persisted document in Promote memory.
   With a team selected, your own personal-workspace datasets are also shown as
   sources when you have direct Read and Share. This explicit promotion path
   does not make personal memory available to normal team searches. Other
   people's personal datasets and transfers between unrelated teams are excluded.
2. Choose Agent → user or User → team, a destination and a reason.
3. Preview the saved document. The API checks source Read and Share, destination
   Write, and the agent-parent or team relationship.
4. Confirm the copy. The revision must match the preview; a changed source must
   be reviewed again. The original remains, and the destination copy records
   its source, actor, reason and previous promotion.
5. Open the destination in Brain and run Cognify when you want its graph built.

Promotion copies the whole persisted document, including a whole persisted
session window if selected. It does not automatically extract only a useful
sentence. The preview displays up to 16,000 bytes and explicitly marks longer
content. The operation accepts at most 64 MiB and does not change permissions or
call an LLM. Repeated promotion of the same source revision preserves destination
edits. SDK agents can propose or invoke this workflow under their own grants;
there is no automatic promotion policy installed by this UI.

## Inspect activity

The Activity tab refreshes every ten seconds and shows registered connections,
declared dataset bindings, recent operations, visible promotion copies, and
available conversations and traces. Background launch success is displayed as
“Started in background”; completion requires a completion record. Actor IDs
refer to the account that executed the operation, not the dataset owner.

This is an inspection view over existing SDK records, not an immutable audit
log or a remote terminal for controlling an agent. It shows the latest 100
connections, pages of 50 operation rows, and the latest 50 visible promotions.

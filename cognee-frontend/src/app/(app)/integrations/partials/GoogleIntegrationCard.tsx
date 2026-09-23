"use client";

import { useCallback, useEffect, useRef, useState, type CSSProperties, type ButtonHTMLAttributes, type InputHTMLAttributes } from "react";
import { Group, Loader, Stack, Text } from "@mantine/core";
import Link from "next/link";
import ModalShell from "@/ui/elements/ModalShell";
import { googleApi, googleAuthorizeUrl, googleError, type GoogleConnection, type GoogleProvider, type GoogleResources } from "@/modules/integrations/googleApi";
import { describeOAuthFailure, isFailureOutcome, takeOAuthOutcome } from "@/modules/integrations/oauthOutcome";
import DataSourceCard from "./DataSourceCard";
import ConnectorLogo from "./ConnectorLogo";

// The SDK shell is dark even when Mantine's global scheme is light. Scope the
// readable text tokens to these cards and the portalled modal body.
const CONTENT_STYLE = {
  color: "var(--color-cognee-fg, #EDECEA)",
  "--mantine-color-text": "var(--color-cognee-fg, #EDECEA)",
  "--mantine-color-dimmed": "#aaa6b0",
} as CSSProperties;

const BUTTON = "cursor-pointer rounded-lg border-none bg-cognee-purple px-3.5 py-1.5 text-white transition-colors hover:bg-cognee-purple-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 disabled:cursor-not-allowed disabled:opacity-40";
function Button({ variant, color, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: string; color?: string }) {
  const secondary = variant === "subtle" || variant === "light";
  return <button {...props} type="button" className={secondary ? "cursor-pointer rounded-lg border border-white/15 bg-white/[0.04] px-3.5 py-1.5 text-cognee-lavender hover:bg-white/[0.08] focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 disabled:cursor-not-allowed disabled:opacity-40" : BUTTON}
    style={{ fontSize: 13, fontWeight: 600, ...(color === "red" ? { color: "#FF8A8A", background: "rgba(239,68,68,.1)" } : {}) }} />;
}
function Checkbox({ label, description, ...props }: InputHTMLAttributes<HTMLInputElement> & { label: string; description?: string | null }) {
  return <label className="flex items-start gap-2.5 text-[13px] text-[var(--color-cognee-fg,#EDECEA)]">
    <input {...props} aria-label={label} type="checkbox" className="mt-0.5 shrink-0 cursor-pointer accent-cognee-lavender disabled:cursor-not-allowed" style={{ width: 14, height: 14 }} />
    <span>{label}{description && <span className="mt-0.5 block text-[11px] text-white/55">{description}</span>}</span>
  </label>;
}

export default function GoogleIntegrationCard({ provider, connecting, setConnecting }: {
  provider: GoogleProvider;
  connecting: GoogleProvider | null;
  setConnecting: (provider: GoogleProvider | null) => void;
}) {
  const isDrive = provider === "google_drive";
  const name = isDrive ? "Google Drive" : "Gmail";
  const resourceName = isDrive ? "folders and shared drives" : "labels";
  const [connection, setConnection] = useState<GoogleConnection>();
  const [statusError, setStatusError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [opened, setOpened] = useState(false);
  const [resourceError, setResourceError] = useState<string | null>(null);
  const [resources, setResources] = useState<GoogleResources>();
  const [selected, setSelected] = useState<string[] | null>([]);
  const [busy, setBusy] = useState(false);
  const [deleteData, setDeleteData] = useState(false);
  const [confirmDisconnect, setConfirmDisconnect] = useState(false);
  const popupPoll = useRef<ReturnType<typeof setInterval> | null>(null);
  const popupWindow = useRef<Window | null>(null);
  const mounted = useRef(false);

  const refreshStatus = useCallback(async () => {
    try {
      const result = await googleApi.connection(provider);
      if (mounted.current) { setConnection(result); setStatusError(null); }
    } catch (err) {
      if (mounted.current) setStatusError(googleError(err));
    }
  }, [provider]);

  useEffect(() => {
    mounted.current = true;
    void refreshStatus();
    const timer = setInterval(() => void refreshStatus(), 10000);
    return () => {
      mounted.current = false;
      clearInterval(timer);
      if (popupPoll.current) { clearInterval(popupPoll.current); setConnecting(null); }
      popupWindow.current?.close();
    };
  }, [refreshStatus, setConnecting]);

  const loadResources = useCallback(async () => {
    setResources(undefined);
    setError(null);
    setResourceError(null);
    try {
      const result = await googleApi.resources(provider);
      if (mounted.current) { setResources(result); setSelected(result.selected); }
    } catch (err) {
      if (mounted.current) { setError(googleError(err)); setResourceError(googleError(err)); }
    }
  }, [provider]);

  useEffect(() => {
    if (opened && connection?.connected) void loadResources();
  }, [opened, connection?.connected, loadResources]);

  async function connect() {
    const popup = window.open("", "cognee-google-oauth", "width=600,height=760");
    if (!popup) { setError("Allow popups for this site, then try connecting again."); return; }
    popupWindow.current = popup;
    const startedAt = Date.now();
    setConnecting(provider);
    setError(null);
    takeOAuthOutcome(provider);
    try {
      const result = await googleApi.authorize(provider);
      if (!mounted.current || popup.closed) { popup.close(); setConnecting(null); return; }
      popup.location.href = googleAuthorizeUrl(result.authorize_url);
      popupPoll.current = setInterval(() => {
        const outcome = takeOAuthOutcome(provider);
        const timedOut = Date.now() - startedAt > 120000;
        if (!outcome && !popup.closed && !timedOut) return;
        if (popupPoll.current) clearInterval(popupPoll.current);
        popupPoll.current = null;
        setConnecting(null);
        popupWindow.current = null;
        if (outcome || timedOut) popup.close();
        if (timedOut && !outcome) setError("Authorization is taking too long. Try Reconnect again.");
        if (isFailureOutcome(outcome)) setError(describeOAuthFailure(outcome, name));
        void refreshStatus();
        if (outcome === "connected") {
          void loadResources();
          setNotice("Account connected.");
        }
      }, 600);
    } catch (err) {
      popup.close();
      if (mounted.current) {
        setConnecting(null);
        const message = googleError(err);
        setError(message.includes("not configured")
          ? `${name} connection is unavailable because this server has not been configured for Google sign-in. Contact the server administrator.`
          : message);
      }
    }
  }

  async function act(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try { await action(); } catch (err) { setError(googleError(err)); }
    finally { if (mounted.current) setBusy(false); }
  }

  const syncing = connection?.sync_status === "syncing";
  const syncIssue = connection?.sync_status === "degraded" || !!resourceError;
  const syncSummary = syncing ? "Syncing…" : resourceError ? "Folder list unavailable" : syncIssue ? "Sync needs attention" : connection?.stored_items !== undefined ? `${connection.stored_items} items stored` : "Ready to sync";
  const dirty = resources !== undefined && JSON.stringify(selected) !== JSON.stringify(resources.selected);
  const listedIds = new Set(resources?.resources.map((resource) => resource.id));
  // Preserve selected IDs absent from a partial/provider-filtered listing.
  const missingIds = (selected ?? []).filter((id) => !listedIds.has(id));
  const options = [...(resources?.resources ?? []), ...missingIds.map((id) => ({
    id, name: id, description: "Previously selected; not returned by Google.", selected: true,
  }))];
  const cancelAuthorization = () => {
    if (popupPoll.current) clearInterval(popupPoll.current);
    popupPoll.current = null;
    popupWindow.current?.close();
    popupWindow.current = null;
    if (connecting === provider) setConnecting(null);
  };
  const close = () => { cancelAuthorization(); setOpened(false); };
  const title = connection?.connected ? `${name} connected` : `Connect ${name}`;
  const permissions = isDrive
    ? ["Read files from the folders and shared drives you select", "Keep your imported files searchable in Cognee"]
    : ["Read messages matching the labels you select", "Keep your imported messages searchable in Cognee"];

  return (
    <>
      <DataSourceCard
        cfg={{
          key: provider, name,
          description: `Read selected ${resourceName} into your Cognee memory.`,
          initials: isDrive ? "Dr" : "Gm",
          logo: isDrive ? "googledrive" : "gmail",
          color: isDrive ? "#4285F4" : "#EA4335",
          permissions: [], supportsChannelRouting: false,
        }}
        state={statusError ? { status: "unavailable" } : connection ? {
          status: connection.connected ? "connected" : "disconnected",
          syncStatus: syncIssue ? "degraded" : undefined,
          lastSyncedAt: connection.last_synced_at ?? undefined,
        } : undefined}
        healthText={syncSummary}
        needsReconnect={false}
        channels={undefined}
        isOwner={true}
        onManageClick={() => {
          setError(null); setNotice(null); setDeleteData(false); setConfirmDisconnect(false); setOpened(true);
        }}
        onRetry={() => void refreshStatus()}
      />
      {opened && <ModalShell label={title} width={420} onClose={close}>
        <div style={{ ...CONTENT_STYLE, display: "flex", flexDirection: "column", gap: 16 }}>
        <div className="flex items-center gap-3">
          <ConnectorLogo logo={isDrive ? "googledrive" : "gmail"} initials={isDrive ? "Dr" : "Gm"} color={isDrive ? "#4285F4" : "#EA4335"} size={38} />
          <h2 className="m-0 flex-1 text-[16px] font-semibold text-[var(--color-cognee-fg,#EDECEA)]">{title}</h2>
          <button onClick={close} aria-label="Close"
            className="cursor-pointer rounded border-none bg-transparent p-0.5 text-[var(--color-cognee-fg,#EDECEA)]/55 transition-colors hover:text-[var(--color-cognee-fg,#EDECEA)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70"
            style={{ fontSize: 18, lineHeight: 1 }}>×</button>
        </div>
        {(error || statusError) && <div role="alert" className="rounded-lg border border-[var(--color-cognee-danger,#EF4444)]/30 bg-[var(--color-cognee-danger,#EF4444)]/10 px-3.5 py-3">
          <p className="m-0 text-[12px] leading-[1.5] text-[var(--color-cognee-danger-fg,#FF8A8A)]">{error || statusError}</p>
        </div>}
        {notice && <Text size="sm" role="status">{notice}</Text>}
        {!connection?.connected ? (connecting === provider ? (
          <p className="m-0 flex items-center gap-2 text-[13px] text-[var(--color-cognee-fg,#EDECEA)]/55">
            <Loader size={14} color="#BC9BFF" />Waiting for authorization in the {name} window…
          </p>
        ) : <div>
          <p className="m-0 mb-3.5 text-[13px] leading-[1.5] text-[var(--color-cognee-fg,#EDECEA)]/55">Cognee will be able to:</p>
          <ul className="m-0 mb-[18px] flex list-none flex-col gap-[7px] p-0">
            {permissions.map((permission) => <li key={permission} className="relative pl-[18px] text-[13px] text-[var(--color-cognee-fg,#EDECEA)]">
              <span className="absolute left-0 text-[var(--color-cognee-success,#22C55E)]">✓</span>{permission}
            </li>)}
          </ul>
          <button disabled={!!connecting} onClick={() => void connect()}
            className="w-full cursor-pointer rounded-lg border-none bg-cognee-purple py-2.5 text-white transition-colors hover:bg-cognee-purple-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 focus-visible:ring-offset-2 focus-visible:ring-offset-black"
            style={{ fontSize: 13, fontWeight: 600 }}>Continue with {name}</button>
          <p className="m-0 mt-2.5 text-center text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">Opens Google’s authorization window — this page stays open</p>
          <p className="m-0 mt-2.5 text-center text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55">Nothing is imported until you select {resourceName} and sync. Imported content is processed by your configured Cognee models.</p>
        </div>) : <Stack gap="md">
          <Text size="sm" c="dimmed">{connection.account_label ?? "Google account"}</Text>
          <section aria-label={isDrive ? "Choose folders" : "Choose labels"}>
            <Text size="sm" fw={600} mb="sm">Choose {isDrive ? "folders" : "labels"}</Text>
            {!resources ? (resourceError ? <Button variant="light" onClick={() => void loadResources()}>Retry resource list</Button> : <Loader size="sm" color="#BC9BFF" />) : <Stack gap="sm">
              <Checkbox label={isDrive ? "Entire My Drive and all shared drives" : "Entire mailbox (all labels)"} checked={selected === null} disabled={busy} onChange={(event) => setSelected(event.currentTarget.checked ? null : [])} />
              <div style={{ maxHeight: 180, overflowY: "auto" }}>
                <Stack gap="xs">
                  {options.map((resource) => <Checkbox key={resource.id} label={resource.name} description={resource.description} disabled={busy || selected === null} checked={selected === null || selected.includes(resource.id)} onChange={(event) => {
                    const checked = event.currentTarget.checked;
                    setSelected((previous) => checked ? [...(previous ?? []), resource.id] : (previous ?? []).filter((id) => id !== resource.id));
                  }} />)}
                  {!options.length && <Text size="sm" c="dimmed">No {resourceName} found.</Text>}
                </Stack>
              </div>
              {!isDrive && <Text size="xs" c="dimmed">Messages must match all selected labels.</Text>}
              <Group>
                <Button disabled={busy || !dirty} onClick={() => void act(async () => {
                  const result = await googleApi.select(provider, selected);
                  setResources({ ...resources, selected: result.selected });
                  setSelected(result.selected);
                  setNotice(syncing ? "Selection saved for the next sync." : "Selection saved.");
                })}>Save selection</Button>
              </Group>
            </Stack>}
          </section>
          <section aria-label="Sync status" className="border-t border-white/10 pt-3">
            <Group justify="space-between" mb="xs">
              <Text size="sm" fw={600}>Sync status: {syncing ? "Syncing…" : connection.sync_status === "ok" ? "Up to date" : syncIssue ? "Needs attention" : "Not synced"}</Text>
              <Button disabled={busy || syncing || !resources || dirty || selected?.length === 0} onClick={() => void act(async () => {
                const result = await googleApi.sync(provider);
                if (!result.accepted) throw new Error("The server did not accept the sync request.");
                await refreshStatus();
                setNotice("Sync started.");
              })}>Refresh</Button>
            </Group>
            {syncing && <Text size="xs" c="dimmed">Processing in the background. You can close this dialog.</Text>}
            {dirty && <Text size="xs" c="dimmed">Save your selection before refreshing.</Text>}
            {resources && selected?.length === 0 && <Text size="xs" c="dimmed">Select {resourceName} to start syncing.</Text>}
            <Text size="xs" c="dimmed">Last sync: {connection.last_synced_at ? new Date(connection.last_synced_at).toLocaleString() : "Never"}</Text>
            {connection.stored_items !== undefined && <Text size="xs" c="dimmed">{connection.stored_items} items stored</Text>}
            {connection.dataset_id && <Link href={`/datasets/${connection.dataset_id}`} className="text-[12px] text-cognee-lavender underline">View in Brain</Link>}
            {connection.sync_status === "degraded" && <Text c="yellow" size="xs" mt="xs">{connection.sync_counts?.failed_rate_limit ? "Google rate limit reached. Wait a minute, then Refresh." : "Some content could not be processed. Refresh to retry."}</Text>}
            {connection.sync_counts && <details className="mt-2 text-[12px] text-white/55">
              <summary className="cursor-pointer">Sync details</summary>
              <div aria-label="Sync counts" className="mt-2">
                <p className="my-1">Scanned items are not necessarily stored items.</p>
                {Object.entries(connection.sync_counts).map(([key, value]) => <div key={key}>{key.replaceAll("_", " ")}: {value}</div>)}
              </div>
            </details>}
          </section>
          <section aria-label="Account actions" className="border-t border-white/10 pt-3">
            {connecting === provider ? <Group justify="space-between">
              <Text size="xs" role="status">Waiting for Google…</Text>
              <Button variant="subtle" onClick={cancelAuthorization}>Cancel</Button>
            </Group> : !confirmDisconnect ? <Group justify="space-between">
              <Button variant="subtle" disabled={!!connecting || busy || syncing} onClick={() => void connect()}>Reconnect</Button>
              <Button color="red" variant="subtle" disabled={busy || syncing} onClick={() => setConfirmDisconnect(true)}>Disconnect</Button>
            </Group> : <Stack gap="sm">
              <Text size="sm">Disconnect this account? Imported data is kept unless you choose deletion.</Text>
              <Checkbox label="Also delete this account’s imported Cognee dataset" checked={deleteData} disabled={busy} onChange={(event) => setDeleteData(event.currentTarget.checked)} />
              <Group>
                <Button color="red" disabled={busy} onClick={() => void act(async () => {
                  try {
                    await googleApi.disconnect(provider, deleteData);
                    setOpened(false);
                  } finally {
                    await refreshStatus();
                  }
                })}>Confirm disconnect</Button>
                <Button variant="subtle" disabled={busy} onClick={() => setConfirmDisconnect(false)}>Cancel</Button>
              </Group>
            </Stack>}
          </section>
        </Stack>}
        </div>
      </ModalShell>}
    </>
  );
}

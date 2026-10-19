"use client";

import { useEffect, useState } from "react";
import { request, describeError, type WorkspaceContext } from "@/modules/workspace/api";
import styles from "../workspace.module.css";

interface Connection { connected: boolean; accountLabel?: string; providerAccountId?: string }
interface Channel { id: string; name: string; allowed: boolean }

function Source({ provider }: { provider: WorkspaceContext["providers"][number] }) {
  const [connection, setConnection] = useState<Connection | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [channels, setChannels] = useState<Channel[] | null>(null);
  const [restrict, setRestrict] = useState(false);
  const name = provider.provider === "slack" ? "Slack" : "GitHub";
  useEffect(() => {
    let cancelled = false;
    request<Connection>(`/v1/integrations/${provider.provider}/connection`)
      .then((value) => { if (!cancelled) setConnection(value); })
      .catch((error) => { if (!cancelled) setError(describeError(error)); });
    return () => { cancelled = true; };
  }, [provider.provider]);
  async function run(action: () => Promise<void>) {
    setBusy(true); setError("");
    try { await action(); } catch (error) { setError(describeError(error)); }
    finally { setBusy(false); }
  }
  return <section className={`${styles.card} ${styles.stack}`}>
    <div className={styles.row}><h2>{name}</h2><span className={styles.badge}>{connection === null ? "Checking" : connection.connected ? "Connected" : "Not connected"}</span></div>
    <p>{name === "Slack" ? "Save selected messages and notes with the existing Cognee Slack app. Channel settings below control where slash commands can run; they do not import channel history." : "Index code from the repositories selected in your GitHub App installation. Repository access stays limited by that installation."}</p>
    {connection?.accountLabel && <strong>{connection.accountLabel}</strong>}
    {error && <div role="alert" className={styles.error}>{error}</div>}
    {!provider.configured && <details><summary>Server setup required</summary><p>Configure your own {name} app on this server. Required settings:</p><ul>{provider.missing_settings.map((key) => <li key={key}><code>{key}</code></li>)}</ul></details>}
    <div className={styles.row}>
      {!connection?.connected && <button className={`${styles.button} ${styles.primary}`} disabled={busy || !provider.configured || connection === null} onClick={() => run(async () => {
        const value = await request<{ authorizeUrl: string }>(`/v1/integrations/${provider.provider}/authorize`, "POST");
        const url = new URL(value.authorizeUrl);
        if (url.protocol !== "https:" || !["slack.com", "github.com"].includes(url.hostname)) throw new Error("The server returned an unexpected authorization address");
        window.location.assign(url.href);
      })}>Connect {name}</button>}
      {connection?.connected && <button className={styles.button} disabled={busy} onClick={() => {
        if (window.confirm(`Disconnect ${name}? Existing Cognee memory will remain.`)) void run(async () => {
          await request(`/v1/integrations/${provider.provider}/connection`, "DELETE");
          setConnection({ connected: false }); setChannels(null);
        });
      }}>Disconnect</button>}
      {connection?.connected && name === "GitHub" && <a className={styles.link} target="_blank" rel="noreferrer" href="https://github.com/settings/installations">Choose repositories on GitHub</a>}
      {connection?.connected && name === "Slack" && <button className={styles.button} disabled={busy} onClick={() => run(async () => {
        const value = await request<{ channels: Channel[] }>("/v1/slack/channels");
        setChannels(value.channels); setRestrict(value.channels.some((channel) => channel.allowed));
      })}>Manage command channels</button>}
    </div>
    {channels && <div className={styles.stack}>
      <label><input type="checkbox" checked={restrict} onChange={(e) => setRestrict(e.target.checked)} /> Restrict slash commands to selected channels</label>
      {restrict && channels.map((channel) => <label key={channel.id}><input type="checkbox" checked={channel.allowed} onChange={(e) => setChannels(channels.map((row) => row.id === channel.id ? { ...row, allowed: e.target.checked } : row))} /> #{channel.name}</label>)}
      <button className={styles.button} disabled={busy || (restrict && !channels.some((row) => row.allowed))} onClick={() => run(async () => {
        await request("/v1/slack/channels", "PUT", { channel_ids: restrict ? channels.filter((row) => row.allowed).map((row) => row.id) : [] });
        setChannels(null);
      })}>Save channel settings</button>
    </div>}
  </section>;
}

export default function Sources({ context }: { context: WorkspaceContext }) {
  return <div className={styles.grid}>{context.providers.map((provider) => <Source key={provider.provider} provider={provider} />)}</div>;
}

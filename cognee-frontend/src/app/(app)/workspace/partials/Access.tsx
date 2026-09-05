"use client";

import { useEffect, useState } from "react";
import { request, describeError, type Access as AccessState, type Permission, type WorkspaceContext } from "@/modules/workspace/api";
import styles from "../workspace.module.css";

const PERMISSIONS: Permission[] = ["read", "write", "share", "delete"];

export default function Access({ context, refresh }: { context: WorkspaceContext; refresh: () => Promise<void> }) {
  const [dataset, setDataset] = useState("");
  const [access, setAccess] = useState<AccessState | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [plugin, setPlugin] = useState("codex");
  const [credential, setCredential] = useState<{ agentId: string; apiKey: string } | null>(null);
  const [identitiesChanged, setIdentitiesChanged] = useState(0);
  useEffect(() => {
    let cancelled = false;
    setAccess(null); setError("");
    if (dataset) request<AccessState>(`/v1/workspace/datasets/${dataset}/access`)
      .then((value) => { if (!cancelled) setAccess(value); })
      .catch((error) => { if (!cancelled) setError(describeError(error)); });
    return () => { cancelled = true; };
  }, [dataset, identitiesChanged]);
  async function change(principalId: string, permission: Permission, allowed: boolean) {
    setBusy(true); setError("");
    try {
      setAccess(await request<AccessState>(`/v1/workspace/datasets/${dataset}/access`, "PUT", { principal_id: principalId, permission, allowed }));
      await refresh();
    } catch (error) { setError(describeError(error)); }
    finally { setBusy(false); }
  }
  return <div className={styles.stack}>
    <div className={styles.card}><h2>Dataset permissions</h2>
      <p>Read allows searching. Write allows saving and processing. Share allows granting access and promoting saved documents. Delete allows removing data. Inherited access comes from team or role grants.</p>
      <label className={styles.field}>Dataset<select value={dataset} onChange={(e) => setDataset(e.target.value)} disabled={busy}>
        <option value="">Choose a dataset you can share</option>
        {context.datasets.filter((row) => row.permissions.includes("share")).map((row) => <option key={row.id} value={row.id}>{row.name} · {row.id.slice(0, 8)}</option>)}
      </select></label>
      {error && <div role="alert" className={styles.error}>{error}</div>}
      {access && <div className={styles.scroll}><table className={styles.table}>
        <thead><tr><th>Person, agent or group</th>{PERMISSIONS.map((p) => <th key={p}>Direct {p}</th>)}<th>Inherited</th></tr></thead>
        <tbody>{access.principals.map((principal) => <tr key={principal.id}>
          <td>{principal.name}<div className={styles.muted}>{principal.kind}{principal.owner && " · dataset owner"}</div></td>
          {PERMISSIONS.map((permission) => <td key={permission}><input type="checkbox" aria-label={`${permission} for ${principal.name}`} checked={principal.direct.includes(permission)} disabled={busy || principal.owner}
            onChange={(e) => change(principal.id, permission, e.target.checked)} /></td>)}
          <td>{principal.inherited.join(", ") || "None"}</td>
        </tr>)}</tbody>
      </table><p>Removing a direct grant does not remove access inherited from a team or role. Change that group’s grant to remove inherited access. Saved permissions are reloaded after each change.</p></div>}
    </div>
    {!context.user.is_agent && <div className={styles.card}><h2>Set up a plugin identity</h2>
      <p>Create a separate identity before granting agent access. Existing plugin identities are kept; this action never rotates their keys.</p>
      <div className={styles.row}><label className={styles.field}>Plugin<select value={plugin} onChange={(e) => setPlugin(e.target.value)} disabled={busy}><option value="codex">Codex</option><option value="claude-code">Claude Code</option><option value="mcp">MCP</option></select></label>
        <button className={styles.button} disabled={busy} onClick={async () => {
          setBusy(true); setError(""); setCredential(null);
          try { setCredential(await request(`/v1/integrations/plugins/${plugin}/provision?create_only=true`, "POST")); setIdentitiesChanged((value) => value + 1); await refresh(); }
          catch (error) { setError(describeError(error)); } finally { setBusy(false); }
        }}>Create identity</button>
        <button className={styles.button} disabled={busy} onClick={async () => {
          if (!window.confirm(`Revoke ${plugin}'s keys and disconnect it? Its saved memory will remain.`)) return;
          setBusy(true); setError(""); setCredential(null);
          try { await request(`/v1/integrations/plugins/${plugin}`, "DELETE"); setIdentitiesChanged((value) => value + 1); await refresh(); }
          catch (error) { setError(describeError(error)); } finally { setBusy(false); }
        }}>Disconnect plugin</button>
      </div>
      {credential && <div className={styles.stack}><p>Copy this key now. It will not be shown again. In the plugin’s Manage Access skill, choose reconnect and provide this key through its named environment variable.</p>
        <label className={styles.field}>New agent key<input type="password" readOnly value={credential.apiKey} autoComplete="off" /></label>
        <button className={styles.button} onClick={() => navigator.clipboard.writeText(credential.apiKey)}>Copy key</button>
        <button className={styles.button} onClick={() => setCredential(null)}>Hide key</button>
      </div>}
    </div>}
  </div>;
}

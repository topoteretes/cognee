"use client";

import { useEffect, useState } from "react";
import { request, describeError, type WorkspaceContext } from "@/modules/workspace/api";
import styles from "../workspace.module.css";

interface Invite { id: string; email: string; expires_at: string; accepted_at: string | null; revoked_at: string | null }
interface Member { id: string; email: string; roles: { id: string; name: string }[] }

export default function Team({ context, refresh }: { context: WorkspaceContext; refresh: () => Promise<void> }) {
  const team = context.teams.find((row) => row.id === context.user.tenant_id);
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [token, setToken] = useState("");
  const [issued, setIssued] = useState("");
  const [invites, setInvites] = useState<Invite[]>([]);
  const [members, setMembers] = useState<Member[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  async function reload() {
    if (!team?.is_owner) return;
    const [rows, people] = await Promise.all([
      request<Invite[]>(`/v1/workspace/teams/${team.id}/invitations`),
      request<Member[]>(`/v1/permissions/tenants/${team.id}/users`),
    ]);
    setInvites(rows); setMembers(people);
  }
  useEffect(() => { void reload().catch((error) => setError(describeError(error))); }, [team?.id, team?.is_owner]); // eslint-disable-line react-hooks/exhaustive-deps
  async function run(action: () => Promise<void>) {
    setBusy(true); setError(""); setNotice("");
    try { await action(); await reload(); } catch (error) { setError(describeError(error)); }
    finally { setBusy(false); }
  }
  return <div className={styles.stack}>
    {error && <div role="alert" className={styles.error}>{error}</div>}
    {notice && <div role="status" className={styles.success}>{notice}</div>}
    <section className={styles.card}><h2>Your teams</h2><p>Each person signs in with their own account. Joining a team gives access to datasets shared with that team. It does not expose every member’s private data.</p>
      <div className={styles.row}><label className={styles.field}>Active team<select value={context.user.tenant_id ?? ""} disabled={busy} onChange={(e) => run(async () => {
        await request("/v1/permissions/tenants/select", "POST", { tenant_id: e.target.value || null }); window.location.reload();
      })}><option value="">Personal</option>{context.teams.map((row) => <option key={row.id} value={row.id}>{row.name}</option>)}</select></label>
        <label className={styles.field}>New team name<input value={name} onChange={(e) => setName(e.target.value)} maxLength={120} /></label>
        <button className={styles.button} disabled={busy || !name.trim() || context.user.is_agent} onClick={() => run(async () => {
          await request(`/v1/permissions/tenants?${new URLSearchParams({ tenant_name: name.trim() })}`, "POST"); window.location.reload();
        })}>Create team</button>
      </div>
    </section>
    {team?.is_owner && <section className={`${styles.card} ${styles.stack}`}><h2>Invite to {team.name}</h2>
      <p>Create a code for one email address. Share it yourself. It expires after seven days and can be used once. The recipient must create an account or sign in on this same server with that email.</p>
      <div className={styles.row}><label className={styles.field}>Recipient email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} /></label>
        <button className={styles.button} disabled={busy || !email.includes("@")} onClick={() => run(async () => {
          const result = await request<{ token: string }>(`/v1/workspace/teams/${team.id}/invitations`, "POST", { email }); setIssued(result.token); setEmail("");
        })}>Create invitation</button></div>
      {issued && <div className={styles.row}><span className={styles.muted}>Invitation created. Copy it now; it cannot be retrieved later.</span><button className={styles.button} onClick={() => navigator.clipboard.writeText(issued)}>Copy invitation code</button><button className={styles.button} onClick={() => setIssued("")}>Hide</button></div>}
      <div className={styles.scroll}><table className={styles.table}><thead><tr><th>Email</th><th>Status</th><th>Expires</th><th /></tr></thead><tbody>{invites.map((invite) => <tr key={invite.id}><td>{invite.email}</td><td>{invite.accepted_at ? "Accepted" : invite.revoked_at ? "Revoked" : new Date(invite.expires_at) < new Date() ? "Expired" : "Pending"}</td><td>{new Date(invite.expires_at).toLocaleDateString()}</td><td>{!invite.accepted_at && !invite.revoked_at && <button className={styles.button} disabled={busy} onClick={() => run(async () => { await request(`/v1/workspace/teams/${team.id}/invitations/${invite.id}`, "DELETE"); })}>Revoke</button>}</td></tr>)}</tbody></table></div>
      <h2>Members</h2><p>To let someone continue work, grant them Read and Write on the relevant dataset. Add Share if they should manage its access or promote its documents.</p>
      <div className={styles.scroll}><table className={styles.table}><thead><tr><th>Account</th><th>Roles</th><th /></tr></thead><tbody>{members.map((member) => <tr key={member.id}><td>{member.email}</td><td>{member.roles.map((role) => role.name).join(", ") || "Member"}</td><td>{member.id !== context.user.id && <button className={styles.button} disabled={busy} onClick={() => {
        if (window.confirm(`Remove ${member.email} from ${team.name}? Team access will be revoked.`)) void run(async () => { await request(`/v1/permissions/tenants/${team.id}/users/${member.id}`, "DELETE"); });
      }}>Remove</button>}</td></tr>)}</tbody></table></div>
    </section>}
    {!context.user.is_agent && <section className={`${styles.card} ${styles.stack}`}><h2>Accept an invitation</h2><p>Signed in as {context.user.email}. Use the email address named in the invitation.</p>
      <label className={styles.field}>Invitation code<input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} /></label>
      <button className={styles.button} disabled={busy || token.length < 32} onClick={() => run(async () => {
        await request("/v1/workspace/invitations/accept", "POST", { token: token.trim() }); setToken(""); await refresh(); setNotice("Invitation accepted. Select the team above to open its memory.");
      })}>Join team</button>
    </section>}
  </div>;
}

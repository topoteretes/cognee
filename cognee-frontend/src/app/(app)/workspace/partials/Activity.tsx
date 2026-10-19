"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { request, describeError, type WorkspaceContext } from "@/modules/workspace/api";
import styles from "../workspace.module.css";

interface Connection {
  id: string; agent_session_name: string; type: string; status: string;
  user_id: string; session_id: string | null; last_active_at: string | null;
  datasets: { id?: string; name?: string; role: string }[];
}
interface Operation {
  id: string; operation_name: string | null; pipeline_name: string | null;
  outcome: string | null; status: string | null; background: boolean | null;
  owner_email: string | null; owner_id: string | null; user_id: string | null; dataset_name: string | null;
  session_id: string | null; created_at: string; error_class: string | null;
}
interface SessionDetail {
  recent_qas?: { qa_id?: string; question?: string; answer?: string; time?: string }[];
  recent_traces?: unknown[];
}
interface PromotedDocument {
  data_id: string; name: string; dataset_name: string;
  promotion: { reason: string; level: string; promoted_at: string; promoted_by: string };
}
export function operationStatus(row: Pick<Operation, "outcome" | "background" | "status">): string {
  if (row.outcome === "failed" || row.status?.includes("ERRORED")) return "Failed";
  if (row.background && row.outcome === "succeeded") return "Started in background";
  if (row.outcome === "succeeded" || row.status?.includes("COMPLETED")) return "Completed";
  return row.status ? "Running" : "Not recorded";
}
export function operationActor(row: Pick<Operation, "owner_email" | "owner_id" | "user_id">): string {
  if (!row.user_id) return "Not recorded";
  return row.owner_id === row.user_id ? row.owner_email ?? row.user_id : row.user_id;
}

export default function Activity({ context }: { context: WorkspaceContext }) {
  const [connections, setConnections] = useState<Connection[]>([]);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [promotions, setPromotions] = useState<PromotedDocument[]>([]);
  const [dataset, setDataset] = useState("");
  const [offset, setOffset] = useState(0);
  const [live, setLive] = useState(true);
  const [updated, setUpdated] = useState("");
  const [error, setError] = useState("");
  const [detail, setDetail] = useState<{ title: string; data: SessionDetail } | null>(null);
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const params = new URLSearchParams({ limit: "50", offset: String(offset) });
        if (dataset) params.set("dataset_id", dataset);
        const [agents, rows, copies] = await Promise.all([
          request<{ agents: Connection[] }>("/v1/agents/connections?active_only=false&range=all&limit=100"),
          request<Operation[]>(`/v1/activity/pipeline-runs?${params}`),
          request<PromotedDocument[]>("/v1/workspace/promotions"),
        ]);
        if (stopped) return;
        setConnections(agents.agents); setOperations(rows); setPromotions(copies);
        setUpdated(new Date().toLocaleTimeString()); setError("");
      } catch (error) { if (!stopped) setError(describeError(error)); }
      finally { if (!stopped && live) timer = setTimeout(load, 10000); }
    }
    void load();
    return () => { stopped = true; clearTimeout(timer); };
  }, [dataset, offset, live]);
  return <div className={styles.stack}>
    <div className={styles.row}><label><input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} /> Refresh every 10 seconds</label><span className={styles.muted}>{updated ? `Last updated ${updated}` : "Loading activity…"}</span></div>
    {error && <div role="alert" className={styles.error}>Activity refresh failed: {error}. Any displayed rows are from the last successful refresh.</div>}
    <section className={styles.card}><h2>Agent connections</h2><p>Connections and declared dataset bindings reported to this server. Permissions are checked separately on every memory request.</p>
      <div className={styles.scroll}><table className={styles.table}><thead><tr><th>Connection</th><th>Status</th><th>Datasets</th><th>Last active</th><th /></tr></thead><tbody>{connections.map((agent) => <tr key={agent.id}><td>{agent.agent_session_name}<div className={styles.muted}>{agent.type}</div></td><td>{agent.status}</td><td>{agent.datasets.map((row) => `${row.name ?? row.id} (${row.role})`).join(", ") || "None declared"}</td><td>{agent.last_active_at ? new Date(agent.last_active_at).toLocaleString() : "Unknown"}</td><td><button className={styles.button} onClick={async () => {
        setError("");
        try { const data = await request<SessionDetail>(`/v1/agents/connections/${agent.user_id}?${new URLSearchParams({ agent_session_name: agent.agent_session_name })}`); setDetail({ title: agent.agent_session_name, data }); }
        catch (error) { setError(describeError(error)); }
      }}>Inspect</button></td></tr>)}</tbody></table></div>
      {!connections.length && updated && <p>No visible agent connections have been registered.</p>}
    </section>
    <section className={`${styles.card} ${styles.stack}`}><h2>Recent operations</h2>
      <label className={styles.field}>Dataset filter<select value={dataset} onChange={(e) => { setDataset(e.target.value); setOffset(0); }}><option value="">My agents and datasets shared with me</option>{context.datasets.filter((row) => row.permissions.includes("read")).map((row) => <option key={row.id} value={row.id}>{row.name}</option>)}</select></label>
      <div className={styles.scroll}><table className={styles.table}><thead><tr><th>When</th><th>Operation</th><th>Actor</th><th>Dataset</th><th>Result</th></tr></thead><tbody>{operations.map((row) => <tr key={row.id}><td>{new Date(row.created_at).toLocaleString()}</td><td>{row.operation_name ?? row.pipeline_name ?? "Operation"}</td><td>{operationActor(row)}</td><td>{row.dataset_name ?? "Not recorded or not readable"}</td><td>{operationStatus(row)}{row.error_class && <div className={styles.muted}>{row.error_class}</div>}</td></tr>)}</tbody></table></div>
      <div className={styles.row}><button className={styles.button} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>Newer</button><button className={styles.button} disabled={operations.length < 50} onClick={() => setOffset(offset + 50)}>Older</button><Link className={styles.link} href="/sessions">Open session conversations and traces</Link></div>
      <p>A pipeline can produce several progress rows. “Started in background” confirms that work began, not that it finished.</p>
    </section>
    <section className={styles.card}><h2>Recent promotions</h2><p>Copies in datasets you can read, including who promoted them and why.</p>
      <div className={styles.scroll}><table className={styles.table}><thead><tr><th>Document</th><th>Destination</th><th>Promoted by</th><th>Reason</th></tr></thead><tbody>{promotions.map((row) => <tr key={row.data_id}><td>{row.name}</td><td>{row.dataset_name} · {row.promotion.level}</td><td>{row.promotion.promoted_by}</td><td>{row.promotion.reason}</td></tr>)}</tbody></table></div>
    </section>
    {detail && <section className={styles.card}><div className={styles.row}><h2>{detail.title}</h2><button className={styles.button} onClick={() => setDetail(null)}>Close details</button></div><p>Recorded session entries and tool traces visible to your account.</p>
      {detail.data.recent_qas?.map((qa, index) => <article key={qa.qa_id ?? index} className={styles.card}><h3>{qa.question ?? "Saved conversation"}</h3><p className={styles.conversation}>{qa.answer}</p>{qa.time && <time className={styles.muted}>{new Date(qa.time).toLocaleString()}</time>}</article>)}
      {!detail.data.recent_qas?.length && <p>No recent conversations are available for this connection.</p>}
      <details><summary>Recorded tool traces ({detail.data.recent_traces?.length ?? 0})</summary><pre className={styles.preview}>{JSON.stringify(detail.data.recent_traces ?? [], null, 2)}</pre></details>
      <details><summary>Connection details</summary><pre className={styles.preview}>{JSON.stringify(detail.data, null, 2)}</pre></details></section>}
  </div>;
}

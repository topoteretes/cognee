"use client";

import { useEffect, useState } from "react";
import { request, describeError, previewDocument, type Promotion, type WorkspaceContext } from "@/modules/workspace/api";
import styles from "../workspace.module.css";

interface Document { id: string; name: string }
export default function Promote({ context, refresh }: { context: WorkspaceContext; refresh: () => Promise<void> }) {
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [document, setDocument] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [level, setLevel] = useState("user");
  const [reason, setReason] = useState("");
  const [preview, setPreview] = useState<Awaited<ReturnType<typeof previewDocument>> | null>(null);
  const [plan, setPlan] = useState<Promotion | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Promotion | null>(null);
  useEffect(() => { setPreview(null); setPlan(null); setConfirmed(false); setResult(null); }, [source, target, document, reason, level]);
  useEffect(() => {
    let cancelled = false;
    setDocuments([]); setDocument("");
    if (source) request<Document[]>(`/v1/datasets/${source}/data`)
      .then((rows) => { if (!cancelled) setDocuments(rows); })
      .catch((error) => { if (!cancelled) setError(describeError(error)); });
    return () => { cancelled = true; };
  }, [source]);
  const body = { data_id: document, source_dataset_id: source, target_dataset_id: target, level, reason };
  async function check() {
    setBusy(true); setError(""); setPlan(null); setPreview(null); setConfirmed(false);
    try {
      const next = await request<Promotion>("/v1/promote", "POST", { ...body, dry_run: true });
      const content = await previewDocument(source, document);
      if (content.revision !== next.source_revision) throw new Error("The document changed during preview. Load it again.");
      setPlan(next); setPreview(content);
    } catch (error) { setError(describeError(error)); }
    finally { setBusy(false); }
  }
  return <section className={`${styles.card} ${styles.stack}`}><h2>Review a document before sharing it more widely</h2>
    <p>Agent → user copies a document into the agent owner’s dataset. User → team copies it into a dataset already shared with the team. Each copy requires source read and share access, plus destination write access. The original stays in place.</p>
    {error && <div role="alert" className={styles.error}>{error}</div>}
    <div className={styles.row}>
      <label className={styles.field}>Source dataset<select disabled={busy} value={source} onChange={(e) => setSource(e.target.value)}><option value="">Select source</option>{context.datasets.filter((row) => row.permissions.includes("read") && row.permissions.includes("share")).map((row) => <option key={row.id} value={row.id}>{row.name} · {row.id.slice(0, 8)}</option>)}</select></label>
      <label className={styles.field}>Saved document<select disabled={busy} value={document} onChange={(e) => setDocument(e.target.value)}><option value="">Select document</option>{documents.map((row) => <option key={row.id} value={row.id}>{row.name}</option>)}</select></label>
    </div>
    <div className={styles.row}>
      <label className={styles.field}>Promotion<select disabled={busy} value={level} onChange={(e) => setLevel(e.target.value)}><option value="user">Agent → user</option><option value="team">User → team</option></select></label>
      <label className={styles.field}>Destination dataset<select disabled={busy} value={target} onChange={(e) => setTarget(e.target.value)}><option value="">Select destination</option>{context.datasets.filter((row) => row.id !== source && row.permissions.includes("write")).map((row) => <option key={row.id} value={row.id}>{row.name} · {row.id.slice(0, 8)}</option>)}</select></label>
    </div>
    <label className={styles.field}>Why should this memory be shared?<textarea disabled={busy} value={reason} maxLength={2000} onChange={(e) => setReason(e.target.value)} /></label>
    <button className={styles.button} disabled={busy || !source || !target || !document || !reason.trim()} onClick={check}>{busy ? "Checking…" : "Preview and check permissions"}</button>
    {plan && preview && <div className={styles.stack}>
      <div className={styles.muted}>{preview.size.toLocaleString()} bytes · {preview.truncated ? "Preview shows the first 16,000 bytes. The entire document will be copied." : "Complete document shown below."}</div>
      <pre className={styles.preview}>{preview.text}</pre>
      <label><input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} disabled={busy || !!result} /> I have reviewed the document and want to copy all of it to the selected destination.</label>
      <button className={`${styles.button} ${styles.primary}`} disabled={!confirmed || busy || !!result} onClick={async () => {
        setBusy(true); setError("");
        try { setResult(await request<Promotion>("/v1/promote", "POST", { ...body, dry_run: false, expected_source_revision: plan.source_revision })); await refresh(); }
        catch (error) { setError(describeError(error)); } finally { setBusy(false); }
      }}>Confirm promotion</button>
    </div>}
    {result && <div role="status" className={styles.success}>{result.status === "copied" ? "Document copied." : "This document revision was already copied; no duplicate was created."} Open the destination in Brain and run Cognify to make the copy searchable in its knowledge graph.</div>}
  </section>;
}

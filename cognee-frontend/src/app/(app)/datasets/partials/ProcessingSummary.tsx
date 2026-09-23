"use client";

import type { DatasetProcessing } from "@/modules/datasets/useDatasetProcessing";

export default function ProcessingSummary({ data, failed, running, error, onRefresh }: {
  data?: DatasetProcessing;
  failed: boolean;
  running: boolean;
  error: boolean;
  onRefresh: () => void;
}) {
  const percent = data?.total ? Math.round(data.completed / data.total * 100) : 0;
  return (
    <section aria-label="Processing progress" style={{ padding: "14px 16px", borderBottom: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)", flexShrink: 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, justifyContent: "space-between" }}>
        <strong style={{ fontSize: 13, color: "#EDECEA" }}>
          {data ? `${data.completed.toLocaleString()} of ${data.total.toLocaleString()} ready to search` : error ? "Processing progress unavailable" : "Loading processing progress…"}
        </strong>
        <button type="button" onClick={onRefresh} title="Refresh processing progress" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "3px 10px", fontSize: 11, color: "#EDECEA", cursor: "pointer" }}>Refresh</button>
      </div>
      {data && <>
        <div role="progressbar" aria-label="Items ready to search" aria-valuenow={data.completed} aria-valuemin={0} aria-valuemax={data.total || 1} style={{ height: 5, borderRadius: 4, overflow: "hidden", background: "rgba(255,255,255,0.1)", margin: "10px 0" }}>
          <div style={{ height: "100%", width: `${percent}%`, background: "#BC9BFF", transition: "width 250ms" }} />
        </div>
        <div style={{ fontSize: 12, color: "rgba(237,236,234,0.7)" }}>
          {data.pending.toLocaleString()} remaining
          {running ? " · Processing in progress" : failed ? " · Last processing run failed" : data.pending > 0 ? " · Not yet ready to search" : data.total > 0 ? " · All imported items processed" : " · No items imported yet"}
        </div>
      </>}
      <p style={{ fontSize: 11, color: error ? "#FBBF24" : "rgba(237,236,234,0.5)", margin: "6px 0 0" }}>
        {error ? "Couldn’t refresh progress. Showing the last available counts." : "Updates every 5 seconds. Counts cover imported items; more may arrive while your source syncs."}
      </p>
    </section>
  );
}

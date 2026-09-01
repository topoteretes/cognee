"use client";

import React, { useMemo, useState } from "react";
import { TrackPageView } from "@/modules/analytics";
import { useFilter } from "@/ui/layout/FilterContext";
import { channelForSessionId, datasetNameById } from "@/modules/sessions/getSessions";
import { useDashboardTelemetry } from "@/app/(app)/dashboard/hooks/useDashboardTelemetry";
import type { Range } from "@/ui/elements/AgentActivityTerminal";
import { AsciiFrame } from "@/app/(app)/dashboard/partials/redesign/AsciiFrame";
import { FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { RangeToggle, type DashRange } from "@/app/(app)/dashboard/partials/redesign/RangeToggle";
import { fmtTokens } from "@/app/(app)/dashboard/partials/redesign/CostPanel";

interface BreakdownRow { name: string; tokens: number }

const RANGE_MS: Record<DashRange, number> = {
  "24h": 24 * 60 * 60 * 1000,
  "7d": 7 * 24 * 60 * 60 * 1000,
  "30d": 30 * 24 * 60 * 60 * 1000,
};

function BreakdownList({ title, rows }: { title: string; rows: BreakdownRow[] }): React.ReactElement {
  const max = Math.max(...rows.map((r) => r.tokens), 1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1, minWidth: 260 }}>
      <span style={{ ...FONT, fontSize: 13, fontWeight: 600, color: T.text, marginBottom: 6 }}>{title}</span>
      {rows.length === 0 && (
        <span style={{ ...FONT, fontSize: 12, color: T.faint, padding: "8px 0" }}>No usage recorded in this range</span>
      )}
      {rows.map((r, i) => (
        <div key={r.name} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", borderTop: i === 0 ? "none" : `1px solid ${T.frame}` }}>
          <span style={{ ...FONT, fontSize: 13, color: T.text, width: 120, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.name}</span>
          <div style={{ flex: 1, height: 6, borderRadius: 0, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
            <div style={{ width: `${Math.max(2, (r.tokens / max) * 100)}%`, height: "100%", background: T.lavender, borderRadius: 0 }} />
          </div>
          <span style={{ ...FONT, fontSize: 13, color: T.text, width: 64, textAlign: "right", flexShrink: 0, fontVariantNumeric: "tabular-nums" }}>{fmtTokens(r.tokens)}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * Real per-operation tokens, broken down by brain and access channel — the
 * same activity feed CostPanel sums for its headline (see sumMeasuredTokens),
 * not a dollar figure reconstructed from spend. Dollars and tokens stay
 * separate, non-reconciling numbers throughout the dashboard (CLO-600).
 */
export default function AnalyticsPage(): React.ReactElement {
  const [range, setRange] = useState<DashRange>("7d");
  const { datasets } = useFilter();
  const { runs, loading } = useDashboardTelemetry(range as Range);

  const { byDataset, byChannel, total } = useMemo(() => {
    const cutoff = Date.now() - RANGE_MS[range];
    const nameById = datasetNameById(datasets);
    const datasetTokens = new Map<string, number>();
    const channelTokens = new Map<string, number>();
    let totalTokens = 0;

    for (const r of runs) {
      const at = r.started_at ?? r.created_at;
      if (!at) continue;
      const ts = new Date(at).getTime();
      if (Number.isNaN(ts) || ts < cutoff) continue;
      if (r.tokens_in === null && r.tokens_out === null) continue;
      const tokens = (r.tokens_in ?? 0) + (r.tokens_out ?? 0);
      if (tokens <= 0) continue;
      totalTokens += tokens;
      const dataset = r.dataset_id ? (nameById.get(r.dataset_id) ?? r.dataset_id) : "—";
      const channel = r.session_id ? channelForSessionId(r.session_id) : "API";
      datasetTokens.set(dataset, (datasetTokens.get(dataset) ?? 0) + tokens);
      channelTokens.set(channel, (channelTokens.get(channel) ?? 0) + tokens);
    }

    const toRows = (m: Map<string, number>): BreakdownRow[] =>
      [...m.entries()].map(([name, tokens]) => ({ name, tokens })).sort((a, b) => b.tokens - a.tokens);
    return { byDataset: toRows(datasetTokens), byChannel: toRows(channelTokens), total: totalTokens };
  }, [runs, datasets, range]);

  return (
    <div style={{ minHeight: "100%", padding: "clamp(16px, 3vw, 32px)", display: "flex", flexDirection: "column", gap: 20 }}>
      <TrackPageView page="Analytics" />

      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ ...FONT, margin: 0, fontSize: 20, fontWeight: 300, color: T.text }}>Analytics</h1>
          <p style={{ ...FONT, margin: "5px 0 0", fontSize: 13, color: T.muted }}>Where your tokens went, by brain and access channel.</p>
        </div>
        <RangeToggle value={range} onChange={setRange} />
      </div>

      <AsciiFrame label={null}>
        {loading && (
          <div style={{ ...FONT, fontSize: 13, color: T.muted, padding: "4px 0 16px" }}>Loading…</div>
        )}
        {!loading && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <span style={{ ...FONT, fontSize: 22, fontWeight: 700, color: T.lavender, fontVariantNumeric: "tabular-nums" }}>{fmtTokens(total)}</span>
              <span style={{ ...FONT, fontSize: 12, color: T.muted }}>tokens routed through memory in this range</span>
            </div>
            <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
              <BreakdownList title="By brain" rows={byDataset} />
              <BreakdownList title="By access channel" rows={byChannel} />
            </div>
          </div>
        )}
      </AsciiFrame>
    </div>
  );
}

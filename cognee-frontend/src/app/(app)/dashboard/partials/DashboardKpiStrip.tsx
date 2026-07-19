"use client";

import SkeletonBar from "@/ui/elements/SkeletonBar";

interface DashboardKpiStripProps {
  liveAgents: number;
  apiCalls: number;
  sessionCount: number;
  graphNodes: number | null;
  graphEdges: number | null;
  brains: number;
  dataLoading: boolean;
}

interface Metric {
  label: string;
  value: number;
  loading: boolean;
  skeletonWidth: number;
}

/**
 * Horizontal KPI strip with six metrics. Each metric gates on its own
 * loading condition — graph counts arrive on a separate async path and stay
 * as skeleton until both totalNodes and totalEdges are ready, so the number
 * transitions skeleton → final exactly once with no intermediate zero.
 */
export function DashboardKpiStrip({
  liveAgents,
  apiCalls,
  sessionCount,
  graphNodes,
  graphEdges,
  brains,
  dataLoading,
}: DashboardKpiStripProps): React.ReactElement {
  const graphLoading = dataLoading || graphNodes === null || graphEdges === null;

  const metrics: Metric[] = [
    { label: "Agents",      value: liveAgents,      loading: dataLoading,  skeletonWidth: 24 },
    { label: "Sessions",    value: sessionCount,     loading: dataLoading,  skeletonWidth: 36 },
    { label: "API calls",   value: apiCalls,         loading: dataLoading,  skeletonWidth: 36 },
    { label: "Graph nodes", value: graphNodes ?? 0,  loading: graphLoading, skeletonWidth: 48 },
    { label: "Graph edges", value: graphEdges ?? 0,  loading: graphLoading, skeletonWidth: 48 },
    { label: "Brains",      value: brains,           loading: dataLoading,  skeletonWidth: 20 },
  ];

  return (
    <div style={{
      background: "rgba(255,255,255,0.06)",
      backdropFilter: "blur(12px)",
      border: "1px solid rgba(255,255,255,0.1)",
      borderRadius: 12,
      overflow: "hidden",
    }}>
      <div style={{ display: "flex", overflowX: "auto" }}>
        {metrics.map((m, i) => {
          const isZero = !m.loading && m.value === 0;
          return (
            <div key={m.label} style={{ display: "flex", alignItems: "stretch", flex: "1 1 0", minWidth: 88 }}>
              <div style={{ flex: 1, padding: "14px 18px", display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{
                  fontSize: 11,
                  color: "rgba(255,255,255,0.4)",
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  whiteSpace: "nowrap",
                }}>
                  {m.label}
                </span>
                <span style={{
                  fontSize: 22,
                  fontWeight: 700,
                  color: isZero ? "rgba(255,255,255,0.2)" : "#EDECEA",
                  fontVariantNumeric: "tabular-nums",
                  lineHeight: "28px",
                  display: "flex",
                  alignItems: "center",
                  minHeight: 28,
                }}>
                  {m.loading
                    ? <SkeletonBar width={m.skeletonWidth} height={18} />
                    : m.value.toLocaleString()}
                </span>
              </div>
              {i < metrics.length - 1 && (
                <div style={{ width: 1, background: "rgba(255,255,255,0.08)", alignSelf: "stretch", flexShrink: 0 }} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

import type { SdkSyncRun } from "@/modules/integrations/sdkApi";

const PHASE_LABELS: Record<string, string> = {
  queued: "Queued",
  checking: "Checking connection",
  discovering: "Discovering repositories",
  downloading: "Downloading repositories",
  indexing: "Indexing repositories",
  processing: "Processing repositories",
  finalizing: "Finalizing sync",
};

const STATE_LABELS: Record<SdkSyncRun["state"], string> = {
  running: "Syncing",
  waiting: "Waiting",
  succeeded: "Sync complete",
  partial: "Sync partially completed",
  failed: "Sync failed",
  cancelled: "Sync cancelled",
  blocked: "Sync blocked",
};

/** Counters describe the reported phase, not an estimated overall percentage. */
export default function SdkSyncProgress({ runs, activityUnknown }: {
  runs: SdkSyncRun[];
  activityUnknown?: boolean;
}) {
  if (runs.length === 0) return null;
  return <div className="my-3 flex flex-col gap-3">
    {runs.map((run) => {
      const active = !activityUnknown && (run.state === "running" || run.state === "waiting");
      const phase = PHASE_LABELS[run.phase] ?? run.phase;
      const label = activityUnknown ? "Last reported progress — activity unconfirmed"
        : run.state === "running" ? phase : STATE_LABELS[run.state];
      const percent = run.total !== null && run.total > 0 && run.completed !== null
        ? Math.min(100, Math.round(run.completed / run.total * 100))
        : run.state === "succeeded" && run.total === 0 ? 100 : undefined;
      const counts = run.completed !== null
        ? `${run.completed}${run.total !== null ? ` / ${run.total}` : ""} ${run.unit ?? "items"}` : null;
      return <div key={run.runId} className="rounded-lg border border-white/10 bg-white/[0.04] p-3">
        <div className="mb-2 flex justify-between gap-3 text-[13px]">
          <span>{label}</span>
          {percent !== undefined && <span>{percent}%{active ? " of this phase" : ""}</span>}
        </div>
        <div role="progressbar" aria-label="Repository sync progress" aria-valuemin={0} aria-valuemax={100}
          aria-valuenow={percent} aria-valuetext={`${label}${counts ? `: ${counts}` : ""}`}
          className="h-2.5 overflow-hidden rounded-full bg-white/10">
          <div className={`h-full rounded-full ${run.state === "succeeded" ? "bg-emerald-400" : "bg-cognee-lavender"} ${active && percent === undefined ? "animate-pulse" : ""}`}
            style={{ width: percent === undefined ? (active ? "100%" : "0%") : `${percent}%` }} />
        </div>
        {counts && <p className="mb-0 mt-2 text-[12px] text-white/60">{counts}</p>}
        {active && <p className="mb-0 mt-1 text-[12px] text-white/60">Progress updates when the server reports completed work.</p>}
      </div>;
    })}
  </div>;
}

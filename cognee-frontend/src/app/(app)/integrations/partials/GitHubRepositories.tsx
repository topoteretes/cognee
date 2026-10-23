"use client";

import { useState } from "react";
import { useDatasetProcessing } from "@/modules/datasets/useDatasetProcessing";
import { sdkApi, sdkError, type SdkConnection, type SdkResource, type SdkRepositoryProgress } from "@/modules/integrations/sdkApi";

function RepositoryProgress({ name, progress, completed, uncertain }: {
  name: string; progress?: SdkRepositoryProgress; completed: boolean; uncertain: boolean;
}) {
  const ready = progress ? progress.state === "succeeded" : completed;
  const failed = progress?.state === "failed" || progress?.state === "cancelled";
  const phases: Record<string, string> = { queued: "Queued", downloading: "Downloading", processing: "Building graph", finalizing: "Finalizing", complete: "Ready", failed: "Failed", cancelled: "Cancelled" };
  const label = uncertain && progress && !ready && !failed ? "Activity unconfirmed"
    : ready ? "Ready" : progress ? phases[progress.phase] ?? progress.phase : "Not imported";
  const percent = ready ? 100 : progress ? progress.completed / progress.total * 100 : 0;
  return <div className="flex shrink-0 items-center gap-2" title={`${label}${progress && !ready ? ` · ${progress.completed}/${progress.total} stages` : ""}`}>
    <span className="text-[11px] text-white/60">{label}</span>
    <svg width="30" height="30" viewBox="0 0 36 36" role={ready || failed ? "img" : "progressbar"}
      aria-label={`${name}: ${label}`} aria-valuenow={!ready && !failed ? percent : undefined}
      aria-valuemin={!ready && !failed ? 0 : undefined} aria-valuemax={!ready && !failed ? 100 : undefined}
      aria-valuetext={progress ? `${progress.completed} of ${progress.total} stages` : label}
      className={ready ? "text-emerald-400" : failed ? "text-red-400" : "text-cognee-lavender"}>
      <circle cx="18" cy="18" r="15" fill="none" stroke="currentColor" strokeWidth="3" opacity="0.2" />
      <circle cx="18" cy="18" r="15" fill="none" stroke="currentColor" strokeWidth="3" pathLength="100"
        strokeDasharray={`${percent} 100`} transform="rotate(-90 18 18)" className="transition-all" />
      {ready && <path d="m11 18 5 5 9-10" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />}
      {failed && <path d="m13 13 10 10m0-10L13 23" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />}
    </svg>
  </div>;
}

function BranchPicker({ repository, branch, disabled, onBusy, onSaved }: {
  repository: string; branch: string | null; disabled: boolean;
  onBusy: (busy: boolean) => void; onSaved: (branch: string | null) => void;
}) {
  const [branches, setBranches] = useState<string[]>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();
  async function load() {
    setLoading(true); setError(undefined);
    try { setBranches((await sdkApi.branches(repository)).branches); }
    catch (error) { setError(sdkError(error)); }
    finally { setLoading(false); }
  }
  async function save(value: string) {
    onBusy(true); setError(undefined); setNotice(undefined);
    try {
      const result = await sdkApi.selectBranch(repository, value || null);
      onSaved(result.branch);
      setNotice("Branch saved. Refresh to import.");
    } catch (error) { setError(sdkError(error)); }
    finally { onBusy(false); }
  }
  return <div className="mt-2 text-[12px]">
    {branches ? <label className="flex items-center gap-2 text-white/60">Branch
      <select aria-label={`Branch for ${repository}`} value={branch ?? ""} disabled={disabled} onChange={event => void save(event.target.value)}
        className="min-w-0 max-w-full rounded border border-white/20 bg-neutral-900 px-2 py-1 text-white">
        <option value="">Default branch</option>
        {branch && !branches.includes(branch) && <option value={branch}>{branch} (unavailable)</option>}
        {branches.map(name => <option key={name} value={name}>{name}</option>)}
      </select>
    </label> : <button type="button" disabled={disabled || loading} onClick={() => void load()}
      aria-label={`Choose branch for ${repository}`} className="cursor-pointer rounded border border-white/15 px-2 py-1 text-cognee-lavender disabled:opacity-40">
      {loading ? "Loading branches…" : `Branch: ${branch ?? "Default branch"} ▾`}
    </button>}
    {error && <p role="alert" className="mb-0 mt-1 text-red-300">{error}</p>}
    {notice && <p role="status" className="mb-0 mt-1 text-white/60">{notice}</p>}
  </div>;
}

export default function GitHubRepositories({ resources, selected, connection, busy, onSelect, onBusy, onBranchSaved }: {
  resources: SdkResource[]; selected: string[] | null; connection: SdkConnection; busy: boolean;
  onSelect: (repository: string, checked: boolean) => void;
  onBusy: (busy: boolean) => void;
  onBranchSaved: (repository: string, branch: string | null) => void;
}) {
  const { data, isError, refetch } = useDatasetProcessing(connection.dataset_id ?? null);
  const runs = connection.sync_progress ?? [];
  return <div>
    {isError && <p role="alert" className="text-[12px] text-red-300">Could not refresh imported status. <button type="button" onClick={() => void refetch()} className="underline">Retry</button></p>}
    <ul className="m-0 flex max-h-80 list-none flex-col gap-2 overflow-y-auto p-0">
      {resources.map(repo => {
        const source = `https://github.com/${repo.id}.git`;
        const branch = repo.attributes?.branch ?? null;
        const observations = runs.map(run => {
          const item = run.resources?.[source];
          if (item && (item.state === "queued" || item.state === "running") && ["failed", "partial", "blocked", "cancelled"].includes(run.state)) {
            const state = run.state === "cancelled" ? "cancelled" : "failed";
            return { ...item, state, phase: state } as SdkRepositoryProgress;
          }
          return item;
        }).filter(item => item?.branch === branch);
        const progress = observations.find(item => item?.state === "running") ?? observations.find(item => item?.state === "queued") ?? observations[0];
        const completed = !isError && !!data?.items.some(item => item.repoUrl === source && (item.repoBranch ?? null) === branch && item.completed);
        return <li key={repo.id} className="rounded-lg border border-white/10 bg-black/20 p-3">
          <div className="flex items-center justify-between gap-2">
            <label className="flex min-w-0 items-start gap-2 text-[13px]">
              <input type="checkbox" aria-label={repo.name} checked={selected === null || selected.includes(repo.id)} disabled={busy}
                onChange={event => onSelect(repo.id, event.target.checked)} className="mt-0.5 shrink-0 accent-cognee-lavender" />
              <span className="break-all">{repo.name}</span>
            </label>
            <RepositoryProgress name={repo.name} progress={progress} completed={completed} uncertain={!!connection.sync_activity_unknown} />
          </div>
          {!repo.unavailable && <BranchPicker repository={repo.id} branch={branch} disabled={busy || connection.sync_status === "syncing"} onBusy={onBusy}
            onSaved={value => onBranchSaved(repo.id, value)} />}
        </li>;
      })}
    </ul>
  </div>;
}

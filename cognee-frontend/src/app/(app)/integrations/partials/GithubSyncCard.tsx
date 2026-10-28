"use client";

import { useCallback, useEffect, useRef, useState, type ReactElement } from "react";
import { describeOAuthFailure, isFailureOutcome, takeOAuthOutcome } from "@/modules/integrations/oauthOutcome";
import ModalShell from "@/ui/elements/ModalShell";
import {
  githubSyncApi, githubSyncError, type GithubRepository, type GithubRepositorySettings,
  type GithubSyncMode, type GithubSyncSettings, type GithubSyncState,
} from "@/modules/integrations/githubSyncApi";
import SdkSourceCard, { type SdkSourceStatus } from "./SdkSourceCard";
import ConnectorLogo from "./ConnectorLogo";

const MODES: { value: GithubSyncMode; label: string; hint: string }[] = [
  { value: "push", label: "On every push", hint: "Sync a repository when GitHub reports a push to its branch." },
  { value: "scheduled", label: "On a schedule", hint: "Sync repositories every set number of minutes." },
  { value: "manual", label: "Manual only", hint: "Sync only when you press Sync now." },
];
const FIELD = "text-[13px] text-[var(--color-cognee-fg,#EDECEA)]";
const MUTED = "m-0 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]/55";
const BUTTON = "cursor-pointer rounded-lg border-none bg-cognee-purple px-3.5 py-1.5 text-[13px] font-semibold text-white hover:bg-cognee-purple-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 disabled:cursor-not-allowed disabled:opacity-40";
const GHOST = "cursor-pointer rounded-lg border border-white/15 bg-white/[0.04] px-3.5 py-1.5 text-[13px] font-semibold text-cognee-lavender hover:bg-white/[0.08] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cognee-lavender/70 disabled:cursor-not-allowed disabled:opacity-40";

function repositoryCount(settings: GithubSyncSettings): string {
  return settings.repositories ? `${Object.keys(settings.repositories).length} selected` : "all repositories";
}

const INSTALL_TIMEOUT_MS = 300000;
const sameJson = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const parseLines = (value: string) => value.split("\n").map((line) => line.trim()).filter(Boolean);

export default function GithubSyncCard(): ReactElement {
  const [state, setState] = useState<GithubSyncState | null | undefined>();
  const [statusError, setStatusError] = useState<string | null>(null);
  const [opened, setOpened] = useState(false);
  const [draft, setDraft] = useState<GithubSyncSettings | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [intervalText, setIntervalText] = useState("60");
  const [pathText, setPathText] = useState<Record<string, string>>({});
  const mounted = useRef(false);
  const removedConfig = useRef<Record<string, GithubRepositorySettings>>({});

  // One place that makes saved settings the draft, so the form never shows stale input.
  const adopt = useCallback((settings: GithubSyncSettings) => {
    setDraft(settings);
    setIntervalText(String(settings.schedule?.intervalMinutes ?? 60));
    setPathText({});
    removedConfig.current = {};
  }, []);

  const refresh = useCallback(async () => {
    try {
      const result = await githubSyncApi.state();
      if (mounted.current) { setState(result); setStatusError(null); }
      return result;
    } catch (err) {
      if (mounted.current) setStatusError(githubSyncError(err));
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void refresh();
    return () => { mounted.current = false; };
  }, [refresh]);

  async function act(action: () => Promise<string>) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try { setNotice(await action()); } catch (err) { setError(githubSyncError(err)); }
    finally { if (mounted.current) setBusy(false); }
  }

  const [repos, setRepos] = useState<GithubRepository[] | null>(null);
  const [reposError, setReposError] = useState<string | null>(null);
  const popup = useRef<Window | null>(null);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);
  const [connecting, setConnecting] = useState(false);
  useEffect(() => () => {
    if (poll.current) clearInterval(poll.current);
    popup.current?.close();
  }, []);

  async function connect() {
    const win = window.open("", "cognee-github-install", "width=600,height=760");
    if (!win) { setError("Allow popups for this site, then try again."); return; }
    popup.current = win;
    setConnecting(true);
    setError(null);
    takeOAuthOutcome("github");
    let url: string;
    try {
      url = await githubSyncApi.authorize();
    } catch (err) {
      win.close();
      if (mounted.current) { setConnecting(false); setError(githubSyncError(err)); }
      return;
    }
    if (!mounted.current || win.closed) { win.close(); if (mounted.current) setConnecting(false); return; }
    win.location.href = url;
    const startedAt = Date.now();
    const finish = (message: string | null) => {
      if (poll.current) clearInterval(poll.current);
      poll.current = null;
      win.close();
      if (!mounted.current) return;
      setConnecting(false);
      if (message) setError(message);
    };
    // The install is done once the server holds a connection, whether or not the
    // popup could report back (GitHub may sever window.opener).
    poll.current = setInterval(async () => {
      const taken = takeOAuthOutcome("github");
      const outcome = typeof taken === "string" ? taken : taken?.outcome ?? null;
      let current: GithubSyncState | null;
      try {
        current = await githubSyncApi.state();
      } catch (err) {
        finish(githubSyncError(err));
        return;
      }
      if (current) {
        finish(null);
        if (mounted.current) { setState(current); adopt(current.settings); }
      } else if (isFailureOutcome(outcome)) {
        finish(describeOAuthFailure(outcome, "GitHub"));
      } else if (outcome === "connected") {
        finish("GitHub finished the install, but this server does not show the connection yet. Reload the page.");
      } else if (win.closed) {
        finish("The GitHub window was closed before the install finished.");
      } else if (Date.now() - startedAt > INSTALL_TIMEOUT_MS) {
        finish("GitHub did not finish the install in time. Try again.");
      }
    }, 1500);
  }

  const connected = !!state;
  const settings = state?.settings;
  const status: SdkSourceStatus | undefined = statusError ? "unavailable"
    : state === undefined ? undefined
    : !state ? "disconnected"
    : state.syncStatus === "degraded" ? "needs_attention" : "connected";
  const healthText = !settings ? "" : !settings.enabled ? "Sync paused"
    : `${MODES.find((mode) => mode.value === settings.mode)?.label ?? settings.mode}, ${repositoryCount(settings)}`;

  const loadRepos = useCallback(async () => {
    setReposError(null);
    try {
      const result = await githubSyncApi.repositories();
      if (mounted.current) setRepos(result);
    } catch (err) {
      if (mounted.current) setReposError(githubSyncError(err));
    }
  }, []);
  useEffect(() => {
    if (opened && connected && !repos) void loadRepos();
  }, [opened, connected, repos, loadRepos]);

  // A dialog opened while the status was still loading fills in once it lands.
  useEffect(() => {
    if (state && !draft) adopt(state.settings);
  }, [state, draft, adopt]);
  // Keep last and next sync fresh while the dialog is open (a sync runs in the background).
  useEffect(() => {
    if (!opened || !connected) return;
    const timer = setInterval(() => void refresh(), 10000);
    return () => clearInterval(timer);
  }, [opened, connected, refresh]);

  const selection = draft?.repositories ?? null;
  const change = (patch: Partial<GithubSyncSettings>) => setDraft((previous) => previous && { ...previous, ...patch });
  const setRepo = (id: string, patch: GithubRepositorySettings | null) => {
    const next = { ...(selection ?? {}) };
    if (patch === null) {
      removedConfig.current[id] = next[id];
      delete next[id];
    } else {
      next[id] = { ...(next[id] ?? removedConfig.current[id]), ...patch };
    }
    change({ repositories: next });
  };
  const setPaths = (repoId: string, kind: "include" | "exclude", text: string) => {
    setPathText((previous) => ({ ...previous, [`${repoId}:${kind}`]: text }));
    const paths = parseLines(text);
    setRepo(repoId, kind === "include" ? { includePaths: paths.length ? paths : null } : { excludePaths: paths });
  };

  const open = () => { if (settings) adopt(settings); setError(null); setNotice(null); setOpened(true); };
  const dirty = !!draft && !!settings && !sameJson(draft, settings);
  const scheduled = draft?.mode === "scheduled";
  const interval = Number(intervalText);
  const intervalValid = !scheduled || (intervalText.trim() !== "" && Number.isInteger(interval) && interval >= 1);
  const changeInterval = (text: string) => {
    setIntervalText(text);
    change({ schedule: { ...draft?.schedule, intervalMinutes: Number(text), timezone: draft?.schedule?.timezone ?? "UTC" } });
  };
  const setMode = (mode: GithubSyncMode) => change({
    mode,
    schedule: mode === "scheduled" ? { ...draft?.schedule, intervalMinutes: Number(intervalText) || 60, timezone: draft?.schedule?.timezone ?? "UTC" } : draft?.schedule,
    firstSync: mode === "scheduled" ? draft?.firstSync : "immediate",
  });

  // Only a repository that was synced and is now left out is retired; the server still wants
  // an acknowledgement for any change of the selection, so a harmless change sends it itself.
  const savedIds = settings?.repositories ? Object.keys(settings.repositories) : (repos ?? []).map((repo) => repo.id);
  const removedIds = !selection ? [] : savedIds.filter((id) => !(id in selection));
  const removedNames = removedIds.map((id) => repos?.find((repo) => repo.id === id)?.fullName ?? `repository ${id}`);
  const selectionChanged = !sameJson(draft?.repositories ?? null, settings?.repositories ?? null);
  const [pendingSync, setPendingSync] = useState<boolean | null>(null);
  const requestSave = (syncNow: boolean) => removedIds.length ? setPendingSync(syncNow) : void save(syncNow, selectionChanged);
  const save = (syncNow: boolean, acknowledged: boolean) => act(async () => {
    setPendingSync(null);
    if (!draft) return "";
    const saved = await githubSyncApi.save(draft, syncNow, acknowledged);
    setState(saved);
    adopt(saved.settings);
    return saved.syncRequested ? "Saved. Sync started in the background." : "Saved.";
  });

  return (
    <>
      <SdkSourceCard
        name="GitHub"
        description="Sync repositories as a code graph that answers questions with source evidence."
        initials="Gh"
        logo="github"
        color="#24292F"
        status={status}
        healthText={healthText}
        actionLabel={statusError ? "Retry" : connected ? "Manage" : "Connect"}
        onAction={statusError ? () => void refresh() : state === undefined ? () => undefined : open}
      />
      {opened && <ModalShell label="GitHub sync" width={460} onClose={() => setOpened(false)}>
        <div className="flex flex-col gap-4 p-5 text-[var(--color-cognee-fg,#EDECEA)]">
          <div className="flex items-center gap-3">
            <ConnectorLogo logo="github" initials="Gh" color="#24292F" size={38} />
            <h2 className="m-0 flex-1 text-[16px] font-semibold">GitHub sync</h2>
            <button onClick={() => setOpened(false)} aria-label="Close" className="cursor-pointer border-none bg-transparent text-[18px] text-[var(--color-cognee-fg,#EDECEA)]/55">×</button>
          </div>
          {error && <p role="alert" className="m-0 text-[12px] text-[var(--color-cognee-danger-fg,#FF8A8A)]">{error}</p>}
          {notice && <p role="status" className={MUTED}>{notice}</p>}
          {!connected ? (
            <>
              <p className={MUTED}>No GitHub account is connected. Install the GitHub app to choose which repositories sync.</p>
              <button className={BUTTON} disabled={connecting} onClick={() => void connect()}>{connecting ? "Waiting for GitHub…" : "Install GitHub app"}</button>
            </>
          ) : !draft ? <p className={MUTED}>Loading…</p> : <>
            <div className="flex flex-wrap items-center gap-2">
              <span className={`${FIELD} flex-1 font-semibold`}>{settings?.enabled ? "Sync is running" : "Sync is paused"}</span>
              <button className={GHOST} disabled={busy || dirty} onClick={() => void act(async () => {
                if (!settings) return "";
                const saved = await githubSyncApi.save({ ...settings, enabled: !settings.enabled }, false, false);
                setState(saved);
                adopt(saved.settings);
                return saved.settings.enabled ? "Sync resumed." : "Sync paused.";
              })}>{settings?.enabled ? "Pause" : "Resume"}</button>
              <button className={BUTTON} disabled={busy || dirty || !settings?.enabled} onClick={() => void act(async () => {
                if (!settings) return "";
                await githubSyncApi.save(settings, true, false);
                void refresh();
                return "Sync started in the background.";
              })}>Sync now</button>
            </div>
            {dirty && <p className={MUTED}>Save your changes before pausing or syncing.</p>}
            <fieldset className="m-0 flex flex-col gap-2 border-none p-0" disabled={busy}>
              <legend className={`${FIELD} mb-1 font-semibold`}>When to sync</legend>
              {MODES.map((mode) => (
                <label key={mode.value} className={`flex items-start gap-2.5 ${FIELD}`}>
                  <input type="radio" name="github-sync-mode" checked={draft.mode === mode.value} onChange={() => setMode(mode.value)} className="mt-0.5 accent-cognee-lavender" />
                  <span>{mode.label}<span className={`${MUTED} block`}>{mode.hint}</span></span>
                </label>
              ))}
            </fieldset>
            {scheduled && (
              <div className="flex flex-col gap-1">
                <label className={`flex items-center gap-2.5 ${FIELD}`}>
                  Every
                  <input type="number" min={1} value={intervalText} disabled={busy} aria-label="Minutes between syncs"
                    onChange={(event) => changeInterval(event.currentTarget.value)}
                    className="w-20 rounded-lg border border-white/15 bg-white/[0.06] px-2 py-1 text-[13px] text-[var(--color-cognee-fg,#EDECEA)]" />
                  minutes ({draft.schedule?.timezone ?? "UTC"})
                </label>
                {!intervalValid && <p role="alert" className="m-0 text-[12px] text-[var(--color-cognee-danger-fg,#FF8A8A)]">Enter a whole number of minutes, 1 or more.</p>}
              </div>
            )}
            <section aria-label="Repositories" className="flex flex-col gap-2">
              <p className={`${FIELD} m-0 font-semibold`}>Repositories</p>
              <label className={`flex items-center gap-2.5 ${FIELD}`}>
                <input type="checkbox" checked={selection === null} disabled={busy}
                  onChange={(event) => change({ repositories: event.currentTarget.checked ? null : {} })} className="accent-cognee-lavender" />
                All repositories available to the installation
              </label>
              {reposError && <p role="alert" className="m-0 text-[12px] text-[var(--color-cognee-danger-fg,#FF8A8A)]">{reposError}</p>}
              {selection !== null && (!repos ? (reposError
                ? <button className={GHOST} onClick={() => void loadRepos()}>Retry repository list</button>
                : <p className={MUTED}>Loading repositories…</p>) : (
                <div className="flex max-h-72 flex-col gap-2 overflow-y-auto">
                  {!repos.length && <p className={MUTED}>This installation has no repositories.</p>}
                  {repos.map((repo) => {
                    const chosen = selection[repo.id];
                    return (
                      <div key={repo.id} className="flex flex-col gap-1.5 rounded-lg border border-white/10 p-2.5">
                        <label className={`flex items-center gap-2.5 ${FIELD}`}>
                          <input type="checkbox" aria-label={repo.fullName} checked={!!chosen} disabled={busy}
                            onChange={(event) => setRepo(repo.id, event.currentTarget.checked ? {} : null)} className="accent-cognee-lavender" />
                          {repo.fullName}
                        </label>
                        {chosen && <>
                          <input aria-label={`Branch of ${repo.fullName}`} value={chosen.branch ?? ""} placeholder={repo.defaultBranch ?? "default branch"} disabled={busy}
                            onChange={(event) => setRepo(repo.id, { branch: event.currentTarget.value.trim() || null })}
                            className="rounded-lg border border-white/15 bg-white/[0.06] px-2 py-1 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]" />
                          <textarea aria-label={`Include paths of ${repo.fullName}`} rows={2} placeholder="Only these paths, one glob per line (empty: all)" disabled={busy}
                            value={pathText[`${repo.id}:include`] ?? (chosen.includePaths ?? []).join("\n")}
                            onChange={(event) => setPaths(repo.id, "include", event.currentTarget.value)}
                            className="rounded-lg border border-white/15 bg-white/[0.06] px-2 py-1 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]" />
                          <textarea aria-label={`Exclude paths of ${repo.fullName}`} rows={2} placeholder="Skip these paths, one glob per line" disabled={busy}
                            value={pathText[`${repo.id}:exclude`] ?? (chosen.excludePaths ?? []).join("\n")}
                            onChange={(event) => setPaths(repo.id, "exclude", event.currentTarget.value)}
                            className="rounded-lg border border-white/15 bg-white/[0.06] px-2 py-1 text-[12px] text-[var(--color-cognee-fg,#EDECEA)]" />
                        </>}
                      </div>
                    );
                  })}
                </div>
              ))}
            </section>
            <p className={MUTED}>
              Last sync: {state?.lastSyncedAt ? new Date(state.lastSyncedAt).toLocaleString() : "Never"}
              {state?.nextSyncAt ? `. Next: ${new Date(state.nextSyncAt).toLocaleString()}` : ""}
            </p>
            {pendingSync !== null && (
              <div role="alert" className="flex flex-col gap-2 rounded-lg border border-[var(--color-cognee-warning,#F59E0B)]/40 p-3">
                <p className={`${FIELD} m-0`}>Leaving out {removedNames.join(", ")} removes {removedNames.length === 1 ? "its" : "their"} indexed data from Cognee memory. Continue?</p>
                <div className="flex gap-2">
                  <button className={BUTTON} autoFocus disabled={busy} onClick={() => void save(pendingSync, true)}>Save and remove</button>
                  <button className={GHOST} disabled={busy} onClick={() => setPendingSync(null)}>Cancel</button>
                </div>
              </div>
            )}
            <p className={MUTED}>Manages the most recently connected GitHub installation.</p>
            <div className="flex flex-wrap gap-2">
              <button className={BUTTON} disabled={busy || !intervalValid} onClick={() => requestSave(false)}>Save</button>
              <button className={GHOST} disabled={busy || !intervalValid} onClick={() => requestSave(true)}>Save and sync now</button>
              {draft.mode === "scheduled" && (
                <button className={GHOST} disabled={busy || dirty} onClick={() => void act(async () => {
                  const result = await githubSyncApi.runDue();
                  return result.accepted ? "Due syncs started in the background." : "No GitHub connection to sync.";
                })}>Run due syncs</button>
              )}
            </div>
          </>}
        </div>
      </ModalShell>}
    </>
  );
}

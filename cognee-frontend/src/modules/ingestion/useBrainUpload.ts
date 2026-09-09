"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTenant } from "@/modules/tenant/TenantProvider";
import type { CogneeInstance } from "@/modules/instances/types";
import { type RememberOptions } from "@/modules/ingestion/rememberData";
import addFiles, { type AddFilesResponse } from "@/modules/ingestion/addFiles";
import { BATCH_CONCURRENCY, BATCH_TIMEOUT_MS, MAX_FILES_PER_UPLOAD } from "@/modules/ingestion/uploadLimits";
import batchFiles, { totalBytes as sumBytes } from "@/modules/ingestion/batchFiles";
import { IDLE_PROGRESS, type UploadProgress, type UploadStage } from "@/modules/ingestion/uploadProgress";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import {
  deleteSession,
  findResumableSession,
  MAX_STALL_MS,
  newSessionId,
  saveSession,
  updateSession,
  type UploadSession,
} from "@/modules/ingestion/uploadSessionStore";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import { datasetStatusQueryKey } from "@/modules/datasets/useDatasetStatuses";
import { captureException } from "@/utils/monitoring";

// Facts about the in-flight upload, handed to every lifecycle callback so each
// call site can build its own analytics/monitoring payloads and UI without
// recomputing them. durationMs is measured from the moment upload() started.
export interface BrainUploadContext {
  datasetId: string;
  files: File[];
  totalBytes: number;
  fileTypes: string[];
  durationMs: number;
  // Files the backend accepted. Below files.length when a batch failed
  // part-way — those files ARE durable, so no caller may report them as lost.
  filesUploaded: number;
}

export interface BrainUploadParams {
  datasetId: string;
  files: File[];
  // When set, forwarded to rememberData so a brand-new dataset can be named in
  // the same call (the brains list passes this; the detail page does not).
  datasetName?: string;
  options?: RememberOptions;
  // Selection exceeded MAX_FILES_PER_UPLOAD — nothing was uploaded.
  onLimitExceeded?: (files: File[]) => void;
  // Every batch landed. Fires before graph-build polling.
  onUploaded?: (ctx: BrainUploadContext) => void;
  // Polling reached a COMPLETED terminal status.
  onProcessed?: (ctx: BrainUploadContext) => void;
  // A batch failed — the add is incomplete (but earlier batches are durable).
  onUploadError?: (error: unknown, ctx: BrainUploadContext) => void;
  // The add succeeded but the knowledge-graph build failed or timed out.
  onProcessingError?: (error: unknown, ctx: BrainUploadContext) => void;
}

// Kept as the historical alias so existing call sites and their prop types
// (brainsTypes.ts, DocumentsPanel) keep compiling; `progress.stage` is the
// richer signal new UI should read.
export type BrainUploadStage = UploadStage;

// A cancelled run is not a failed one: the session survives and is resumable,
// so it must not reach onUploadError and be reported to the user as a break.
function isAbort(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

export interface UseBrainUploadResult {
  isUploading: boolean;
  stage: BrainUploadStage;
  progress: UploadProgress;
  upload: (params: BrainUploadParams) => Promise<void>;
  // Abort the run in flight (manual or auto-resumed). The persisted session
  // survives, so the upload can be picked up again rather than being lost.
  cancel: () => void;
}

/**
 * Batched, resumable ingestion for the dashboard, the brains list, and the
 * dataset detail page.
 *
 * Three properties this owns, all from CLO-492:
 *
 * 1. **No single blocking request.** The selection is split into bounded
 *    batches (batchFiles.ts) sent with small concurrency, so success never
 *    depends on one connection surviving for minutes. This is what makes 200
 *    files work where 100 in one request timed out.
 * 2. **Real progress.** Bytes are counted as they leave the browser (XHR
 *    upload events via instance.upload), not inferred from a spinner.
 * 3. **Survives refresh.** The pending remainder lives in IndexedDB, so a
 *    reload replays only what never left and reattaches to the rest.
 *
 * The add succeeding and the build succeeding stay separate callbacks — a
 * post-add build failure must never be shown as an upload failure (CLO-219).
 */
export function useBrainUpload(instance: CogneeInstance | null): UseBrainUploadResult {
  const { tenant } = useTenant();
  const queryClient = useQueryClient();
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState<UploadProgress>(IDLE_PROGRESS);
  // Bytes already acknowledged by completed batches. In-flight batch progress
  // is tracked separately and summed on top, so a retried batch cannot double
  // count what it already reported.
  const settledBytes = useRef(0);
  const inFlightBytes = useRef(new Map<number, number>());
  const activeSession = useRef<UploadSession | null>(null);
  // One run at a time. settledBytes/inFlightBytes are shared mutable state, so
  // a manual upload() starting while auto-resume is mid-flight would interleave
  // two runs' byte accounting and report nonsense. Checked and set
  // synchronously — no await between the test and the set — so the guard itself
  // cannot be raced.
  const running = useRef(false);
  const abortRun = useRef<AbortController | null>(null);

  const publish = useCallback((patch: Partial<UploadProgress>) => {
    setProgress((prev) => ({ ...prev, ...patch }));
  }, []);

  const recomputeBytes = useCallback(() => {
    let live = 0;
    for (const bytes of inFlightBytes.current.values()) live += bytes;
    publish({ bytesSent: settledBytes.current + live });
  }, [publish]);

  const invalidateStatuses = useCallback((): void => {
    queryClient.invalidateQueries({ queryKey: datasetStatusQueryKey(tenant?.tenant_id) });
  }, [queryClient, tenant?.tenant_id]);

  /**
   * Poll the background build to a terminal state, then drop the session.
   * Shared by a fresh upload and a resumed one.
   */
  const awaitProcessing = useCallback(
    async (
      session: UploadSession,
      inst: CogneeInstance,
      hooks: Pick<BrainUploadParams, "onProcessed" | "onProcessingError">,
      context: () => BrainUploadContext,
    ): Promise<void> => {
      publish({ stage: "processing" });
      // An update, not a creation — same reason as the per-batch write: this
      // must not recreate a session another tab has already taken over.
      //
      // A null result is not acted on, deliberately. It means the record is
      // already gone, so there is nothing left to orphan and nothing to clean
      // up; the files themselves landed and the build they triggered is real,
      // so this tab keeps polling it for the user watching this page. (An
      // "uploading"-stage record surviving a crash is the *opposite* case —
      // the write succeeded — and MAX_AGE_MS is what collects that.)
      await updateSession({ ...session, stage: "processing", pending: [] });
      try {
        // Build the knowledge graph once, for the whole dataset, now that every
        // batch has landed. This is what the per-batch /remember calls used to
        // do 17 times over — each running the full cognify pipeline and queuing
        // on the dataset's single lock while later batches were still uploading.
        // One incremental cognify covers every file added above (already-built
        // items are skipped), including any that landed in an earlier, resumed
        // run. run_in_background=true, so this returns a pipeline_run_id
        // immediately and pollDatasetStatus tracks it to completion.
        //
        // A failure to *start* the build is a processing error, not an upload
        // error: the files are durable, so the caller must not report data loss
        // — the same contract pollDatasetStatus already has (CLO-219). The
        // dataset detail page's "Retry build" button re-runs exactly this call.
        await cognifyDataset(
          { id: session.datasetId, name: session.datasetName ?? "", data: [], status: "processing" },
          inst,
          session.options,
        );
        await pollDatasetStatus(session.datasetId, inst, {
          intervalMs: 5000,
          onStatus: (status) => publish({ processingStatus: status }),
        });
        invalidateStatuses();
        hooks.onProcessed?.(context());
      } catch (error) {
        invalidateStatuses();
        hooks.onProcessingError?.(error, context());
      } finally {
        await deleteSession(session.id);
        activeSession.current = null;
        running.current = false;
        abortRun.current = null;
        setIsUploading(false);
        setProgress(IDLE_PROGRESS);
      }
    },
    [invalidateStatuses, publish],
  );

  /**
   * Send `files` in batches, updating progress and the persisted session after
   * each one. Returns the files that were accepted; throws only after the
   * session has been updated, so a caller's error path still sees an accurate
   * count of what is durable.
   */
  const sendBatches = useCallback(
    async (
      session: UploadSession,
      files: File[],
      inst: CogneeInstance,
      signal?: AbortSignal,
    ): Promise<void> => {
      const batches = batchFiles(files);
      publish({ batchesTotal: batches.length, batchesCompleted: 0 });

      let cursor = 0;
      let failure: unknown = null;
      // Batch index is the unit of bookkeeping, not file identity: each worker
      // owns a distinct index, so marking one done can never race another, and
      // two selected files sharing a name can't cancel each other out.
      const sent = new Set<number>();
      const pendingAfter = (): File[] => batches.filter((b) => !sent.has(b.index)).flatMap((b) => b.files);
      // Accumulated here rather than appended onto activeSession.current: two
      // workers settling close together both read the session from before the
      // other's `await saveSession` resolved, so a read-modify-write off it
      // loses one of the two ids. Every other field is recomputed from
      // authoritative shared state (`sent`, `settledBytes`), which is why they
      // survive the same interleaving and this one did not.
      const runIds: string[] = [...(session.runIds ?? [])];
      // Set once the store tells us this session is gone — another tab has
      // taken the dataset over. Reported once rather than per batch, since
      // every subsequent write returns null too.
      let ownershipLost = false;

      const worker = async (): Promise<void> => {
        for (;;) {
          if (failure) return;
          // An abort has to leave a trace. Returning quietly here made
          // Promise.all resolve, `failure` stay null, and the caller count
          // every file as accepted — so cancelling an upload fired onUploaded
          // for files that never left the browser.
          if (signal?.aborted) return;
          // `cursor++` is safe despite the concurrent workers: JS is
          // single-threaded and this read-modify-write contains no await, so
          // no other worker can observe or mutate cursor mid-expression.
          // Workers only interleave at await points, of which there are none
          // between the read and the write.
          const batch = batches[cursor++];
          if (!batch) return;

          let response: AddFilesResponse | undefined;
          try {
            // Upload only — no graph build per batch. The whole dataset is
            // cognified once, after every batch has landed (see
            // awaitProcessing). session.options carry the cognify settings and
            // are applied there, not here.
            response = await addFiles(
              session.datasetName
                ? { id: session.datasetId, name: session.datasetName }
                : { id: session.datasetId },
              batch.files,
              inst,
              {
                timeoutMs: BATCH_TIMEOUT_MS,
                signal,
                onProgress: (bytesSent) => {
                  inFlightBytes.current.set(batch.index, bytesSent);
                  recomputeBytes();
                },
              },
            );
          } catch (error) {
            failure = error;
            inFlightBytes.current.delete(batch.index);
            recomputeBytes();
            return;
          }

          // Settle this batch: its bytes move from "in flight" to "sent", and
          // its files leave the pending set so a refresh never resends them.
          inFlightBytes.current.delete(batch.index);
          settledBytes.current += batch.bytes;
          sent.add(batch.index);

          // The run id is the only handle a backend log search has on this
          // batch; it was being thrown away, leaving runIds permanently [].
          if (response?.pipeline_run_id) runIds.push(response.pipeline_run_id);

          const remaining = pendingAfter();
          const updated: UploadSession = {
            ...(activeSession.current ?? session),
            pending: remaining,
            filesCompleted: session.filesTotal - remaining.length,
            bytesCompleted: settledBytes.current,
            runIds: [...runIds],
            // A landed batch is real progress, so the stall clock restarts.
            lastProgressAt: Date.now(),
            updatedAt: Date.now(),
          };
          // updateSession, not saveSession: if another tab has taken this
          // dataset over and deleted our record, a put() would resurrect it and
          // leave two sessions competing. A null means we no longer own the run.
          //
          // We keep uploading on purpose — these files are already in flight and
          // abandoning them would lose work the backend is about to accept. The
          // deliberate trade is that this run is no longer *resumable*: with no
          // durable record, a crash from here on cannot be picked up, and the
          // owning tab's session is the one a reload will find. Reported so the
          // case is visible rather than inferred from missing sessions.
          const persisted = await updateSession(updated);
          if (!persisted && !ownershipLost) {
            ownershipLost = true;
            captureException(
              new Error(
                "Upload session was taken over by another tab; this run continues but is no longer resumable",
              ),
              {
                datasetId: session.datasetId,
                sessionId: session.id,
                filesCompleted: updated.filesCompleted,
                filesTotal: session.filesTotal,
              },
            );
          }
          activeSession.current = persisted ?? updated;

          publish({ filesCompleted: updated.filesCompleted, batchesCompleted: sent.size });
          recomputeBytes();
        }
      };

      await Promise.all(
        Array.from({ length: Math.min(BATCH_CONCURRENCY, batches.length) }, () => worker()),
      );

      if (failure) throw failure;
      // Stopping early because of an abort is not the same as sending
      // everything. Without this the run resolves as a full success.
      if (signal?.aborted) throw new DOMException("Upload cancelled", "AbortError");
    },
    [publish, recomputeBytes],
  );

  const upload = useCallback(
    async (params: BrainUploadParams): Promise<void> => {
      if (!instance) return;
      const { datasetId, files, datasetName, options } = params;

      if (files.length > MAX_FILES_PER_UPLOAD) {
        params.onLimitExceeded?.(files);
        return;
      }

      const refuse = (message: string): void => {
        params.onUploadError?.(new Error(message), {
          datasetId,
          files,
          totalBytes: sumBytes(files),
          fileTypes: files.map((f) => f.type || "unknown"),
          durationMs: 0,
          filesUploaded: 0,
        });
      };

      // A session is looked up by tenant id, so one written before the tenant
      // resolves is stamped with a value no later lookup will ever match — it
      // becomes unreachable the moment the page reloads, taking its pending
      // files with it. Refuse rather than start an upload that cannot be
      // resumed, and say so instead of no-opping on a click.
      if (!tenant?.tenant_id) {
        refuse("Still connecting to your workspace. Try the upload again in a moment.");
        return;
      }

      if (running.current) {
        refuse("An upload is already in progress. Wait for it to finish before starting another.");
        return;
      }
      running.current = true;

      const bytesTotal = sumBytes(files);
      const fileTypes = files.map((f) => f.type || "unknown");
      const startedAt = Date.now();
      let accepted = 0;
      const context = (): BrainUploadContext => ({
        datasetId,
        files,
        totalBytes: bytesTotal,
        fileTypes,
        durationMs: Date.now() - startedAt,
        filesUploaded: accepted,
      });

      settledBytes.current = 0;
      inFlightBytes.current.clear();
      abortRun.current = new AbortController();
      setIsUploading(true);
      setProgress({
        ...IDLE_PROGRESS,
        stage: "uploading",
        filesTotal: files.length,
        bytesTotal,
      });

      // Drop any earlier session for this dataset before starting a new one:
      // two live sessions for one dataset are incoherent, and the stale one
      // would otherwise be picked up by the next page load's auto-resume and
      // replay files this run is about to send again.
      const stale = await findResumableSession(tenant.tenant_id, datasetId);
      if (stale) await deleteSession(stale.id);

      const session = await saveSession({
        id: newSessionId(),
        tenantId: tenant.tenant_id,
        datasetId,
        datasetName,
        options,
        createdAt: startedAt,
        updatedAt: startedAt,
        lastProgressAt: startedAt,
        filesTotal: files.length,
        bytesTotal,
        filesCompleted: 0,
        bytesCompleted: 0,
        pending: files,
        fileTypes,
        runIds: [],
        stage: "uploading",
      });
      activeSession.current = session;

      // Phase 1 — send every batch. Files in landed batches are durable even
      // if a later batch fails, so the error path reports what got through
      // rather than implying the whole selection was lost.
      try {
        await sendBatches(session, files, instance, abortRun.current.signal);
        accepted = files.length;
      } catch (error) {
        accepted = activeSession.current?.filesCompleted ?? 0;
        running.current = false;
        abortRun.current = null;
        // Release the session too, not just the run guard. The record stays in
        // IndexedDB on purpose so the remainder can be picked up — but holding
        // a reference to it here makes the resume effect's
        // `if (... || activeSession.current || ...) return` bail out and skip
        // the very session we kept. awaitProcessing clears this in its finally;
        // the failure path has to as well or the two disagree.
        activeSession.current = null;
        setIsUploading(false);
        setProgress(IDLE_PROGRESS);
        // Cancelling (or navigating away) leaves the session on disk to be
        // resumed — surfacing that as an upload error would show a failure
        // banner for something the user asked for.
        if (!isAbort(error)) params.onUploadError?.(error, context());
        return;
      }

      params.onUploaded?.(context());
      await awaitProcessing(activeSession.current ?? session, instance, params, context);
    },
    [awaitProcessing, instance, sendBatches, tenant?.tenant_id],
  );

  // Reattach on mount: finish an upload the previous page load started.
  useEffect(() => {
    if (!instance || !tenant?.tenant_id) return;
    let cancelled = false;

    (async () => {
      const session = await findResumableSession(tenant.tenant_id);
      // Re-test the guard AFTER the await: a manual upload() may have started
      // while the lookup was in flight, and it owns the shared byte counters.
      if (!session || cancelled || activeSession.current || running.current) return;
      // Claim it in the same synchronous step as the test. Anything awaited
      // between the two — persisting the attempt counter, for instance —
      // reopens exactly the window the re-test above exists to close.
      running.current = true;

      // Bound retrying by how long the session has been stuck, not by how many
      // times a page has loaded. The session is deliberately kept when an
      // upload fails so a transient blip doesn't discard files the user would
      // have to re-select — but a permanently-wedged one must eventually stop
      // being picked up. Counting loads got this wrong in the other direction:
      // a slow upload the user reloads past makes no attempt-count progress
      // while still moving forward, and was deleted for it.
      const stalledFor = Date.now() - (session.lastProgressAt ?? session.createdAt);
      if (stalledFor > MAX_STALL_MS) {
        captureException(
          new Error(
            `Abandoned upload session after ${Math.round(stalledFor / 60_000)} minutes with no progress`,
          ),
          {
            datasetId: session.datasetId,
            filesTotal: session.filesTotal,
            filesCompleted: session.filesCompleted,
            pendingFiles: session.pending.length,
          },
        );
        await deleteSession(session.id);
        running.current = false;
        return;
      }

      // A degraded session could not persist its blobs, so its empty `pending`
      // does NOT mean "everything was sent" — it means the remainder is
      // unrecoverable and the user has to re-select those files. Treating the
      // two alike skipped the upload phase and polled as though all N files had
      // landed, which is silent data loss dressed up as success.
      const unrecoverable =
        session.degraded && session.pending.length === 0
          ? session.filesTotal - session.filesCompleted
          : 0;
      if (unrecoverable > 0) {
        captureException(
          new Error(`Upload session lost ${unrecoverable} unsent file(s) to a storage quota failure`),
          {
            datasetId: session.datasetId,
            filesTotal: session.filesTotal,
            filesCompleted: session.filesCompleted,
          },
        );
      }

      const controller = new AbortController();
      abortRun.current = controller;
      activeSession.current = session;
      settledBytes.current = session.bytesCompleted;
      inFlightBytes.current.clear();
      setIsUploading(true);
      setProgress({
        stage: "resuming",
        filesTotal: session.filesTotal,
        filesCompleted: session.filesCompleted,
        bytesTotal: session.bytesTotal,
        bytesSent: session.bytesCompleted,
        batchesTotal: 0,
        batchesCompleted: 0,
        unrecoverableFiles: unrecoverable,
      });

      const context = (): BrainUploadContext => ({
        datasetId: session.datasetId,
        files: session.pending,
        totalBytes: session.bytesTotal,
        // Restored from the session: a resumed run holds no File objects for
        // the batches that already landed, so deriving this from `pending`
        // reported an empty list and broke analytics grouped by file type.
        fileTypes: session.fileTypes ?? [],
        durationMs: Date.now() - session.createdAt,
        // Read live, not from the record as found: by the time the processing
        // hooks fire the replayed remainder has landed too, and the persisted
        // session is what carries that count.
        filesUploaded: activeSession.current?.filesCompleted ?? session.filesCompleted,
      });

      // Anything still pending never left the browser — replay it. A degraded
      // session has an empty `pending` with files still unsent; there is
      // nothing to replay, but it must not be reported as a finished upload
      // either — `unrecoverable` above carries that through to the UI.
      if (session.pending.length > 0) {
        publish({ stage: "uploading" });
        try {
          await sendBatches(session, session.pending, instance, controller.signal);
        } catch (error) {
          // Leave the session in place: the remainder is still on disk, so the
          // next load can try again rather than losing it.
          //
          // But do not swallow the reason. This path has no caller and so no
          // onUploadError to raise — a bare `catch {}` meant an expired token,
          // a 402 or a server error all looked identical to a network blip:
          // the upload simply stopped, with nothing recorded anywhere. A
          // deliberate cancel is the one case that is not a failure.
          if (!isAbort(error)) {
            captureException(error instanceof Error ? error : new Error(String(error)), {
              datasetId: session.datasetId,
              sessionId: session.id,
              stage: "resume",
              filesCompleted: activeSession.current?.filesCompleted ?? session.filesCompleted,
              filesTotal: session.filesTotal,
              pendingFiles: session.pending.length,
            });
          }
          running.current = false;
          abortRun.current = null;
          activeSession.current = null;
          setIsUploading(false);
          setProgress(IDLE_PROGRESS);
          return;
        }
      }
      if (!cancelled) {
        await awaitProcessing(
          activeSession.current ?? session,
          instance,
          {
            // Same gap as the upload leg above, on the build side: this path
            // has no caller and so no onProcessingError to raise. Empty hooks
            // meant a cognify that refused to start (or a build that errored)
            // on a resumed session was swallowed whole — awaitProcessing's
            // finally then deleted the session, so the files left the UI with
            // nothing recorded anywhere. The files themselves are durable;
            // what was lost was any trace that the graph was never built.
            onProcessingError: (error, ctx) => {
              captureException(error instanceof Error ? error : new Error(String(error)), {
                datasetId: session.datasetId,
                sessionId: session.id,
                stage: "resume-processing",
                filesUploaded: ctx.filesUploaded,
                filesTotal: session.filesTotal,
              });
            },
          },
          context,
        );
      } else {
        // Unmounted mid-resume: awaitProcessing (which clears these in its
        // finally) never runs, so release the guard here or no later upload
        // could ever start.
        running.current = false;
        abortRun.current = null;
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [awaitProcessing, instance, publish, sendBatches, tenant?.tenant_id]);

  // Hand the run over on unmount instead of letting two hook instances drive
  // one session. `running` is a per-instance ref but the session is global, so
  // an un-aborted run keeps sending from the unmounting instance while the next
  // mount resumes the same record — ingesting the remainder twice. Aborting is
  // safe precisely because the session survives: the new mount picks it up and
  // continues, re-sending only the batches that were still in flight.
  //
  // Deliberately its own effect with an empty dep array: folding this into the
  // resume effect's cleanup would abort a healthy upload every time one of that
  // effect's dependencies changed, not just on unmount.
  useEffect(() => {
    return () => abortRun.current?.abort();
  }, []);

  // Stops the in-flight run, including one picked up by auto-resume — which
  // otherwise had no way to be interrupted. The session stays in IndexedDB, so
  // an aborted run is resumable rather than lost.
  const cancel = useCallback((): void => {
    abortRun.current?.abort();
  }, []);

  return { isUploading, stage: progress.stage, progress, upload, cancel };
}

"use client";

import { useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTenant } from "@/modules/tenant/TenantProvider";
import type { CogneeInstance } from "@/modules/instances/types";
import rememberData, { type RememberOptions } from "@/modules/ingestion/rememberData";
import { MAX_FILES_PER_UPLOAD } from "@/modules/ingestion/uploadLimits";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import { datasetStatusQueryKey } from "@/modules/datasets/useDatasetStatuses";

// Facts about the in-flight upload, handed to every lifecycle callback so each
// call site can build its own analytics/monitoring payloads and UI without
// recomputing them. durationMs is measured from the moment upload() started.
export interface BrainUploadContext {
  datasetId: string;
  files: File[];
  totalBytes: number;
  fileTypes: string[];
  durationMs: number;
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
  // rememberData resolved: the files are added. Fires before graph-build polling.
  onUploaded?: (ctx: BrainUploadContext) => void;
  // Polling reached a COMPLETED terminal status.
  onProcessed?: (ctx: BrainUploadContext) => void;
  // rememberData itself failed — the add did not happen.
  onUploadError?: (error: unknown, ctx: BrainUploadContext) => void;
  // The add succeeded but the knowledge-graph build failed or timed out.
  onProcessingError?: (error: unknown, ctx: BrainUploadContext) => void;
}

// "uploading": files are being added (rememberData). "processing": the add
// succeeded and the knowledge-graph build is running (pollDatasetStatus) — this
// is usually the multi-minute part, and callers should label it distinctly
// from "uploading" so users don't think the file transfer itself is stuck.
export type BrainUploadStage = "idle" | "uploading" | "processing";

export interface UseBrainUploadResult {
  isUploading: boolean;
  stage: BrainUploadStage;
  upload: (params: BrainUploadParams) => Promise<void>;
}

/**
 * Shared mechanics for the two-phase brain upload flow used by the dashboard,
 * the brains list, and the dataset detail page: validate file count, add the
 * files (rememberData in background), then poll the knowledge-graph build to a
 * terminal status. The add succeeding and the build succeeding are reported
 * through separate callbacks — a post-add build failure must never be shown as
 * an upload failure (CLO-219). The dataset-status query key is always
 * invalidated once polling settles so every open view reflects the result
 * without waiting for its own next poll tick.
 */
export function useBrainUpload(instance: CogneeInstance | null): UseBrainUploadResult {
  const { tenant } = useTenant();
  const queryClient = useQueryClient();
  const [isUploading, setIsUploading] = useState(false);
  const [stage, setStage] = useState<BrainUploadStage>("idle");

  const upload = useCallback(
    async (params: BrainUploadParams): Promise<void> => {
      if (!instance) return;
      const { datasetId, files, datasetName, options } = params;

      if (files.length > MAX_FILES_PER_UPLOAD) {
        params.onLimitExceeded?.(files);
        return;
      }

      const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
      const fileTypes = files.map((f) => f.type || "unknown");
      const startedAt = Date.now();
      const context = (): BrainUploadContext => ({
        datasetId,
        files,
        totalBytes,
        fileTypes,
        durationMs: Date.now() - startedAt,
      });

      const invalidateStatuses = (): void => {
        queryClient.invalidateQueries({ queryKey: datasetStatusQueryKey(tenant?.tenant_id) });
      };

      setIsUploading(true);
      setStage("uploading");

      // Phase 1 — add the files. Once this resolves the add is durable; any
      // later failure is a build failure, not an upload failure.
      try {
        const target = datasetName ? { id: datasetId, name: datasetName } : { id: datasetId };
        await rememberData(target, files, instance, { ...options, runInBackground: true });
      } catch (error) {
        setIsUploading(false);
        setStage("idle");
        params.onUploadError?.(error, context());
        return;
      }

      params.onUploaded?.(context());
      setStage("processing");

      // Phase 2 — poll the background knowledge-graph build to a terminal state.
      try {
        await pollDatasetStatus(datasetId, instance, { intervalMs: 5000 });
        invalidateStatuses();
        params.onProcessed?.(context());
      } catch (error) {
        invalidateStatuses();
        params.onProcessingError?.(error, context());
      } finally {
        setIsUploading(false);
        setStage("idle");
      }
    },
    [instance, queryClient, tenant?.tenant_id],
  );

  return { isUploading, stage, upload };
}

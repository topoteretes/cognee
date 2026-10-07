"use client";

import { describeProgress, uploadFraction, type UploadProgress } from "@/modules/ingestion/uploadProgress";

const ACCENT = "#6510F4";

/**
 * Determinate while bytes are moving; indeterminate for the estimate and the
 * graph build, which have no meaningful percentage.
 *
 * Estimate and build are deliberately NOT a fake percentage: neither the cost
 * estimate nor the backend's add/cognify exposes in-flight progress (CLO-557),
 * so a moving number there would be invented. They pulse with a live stage
 * label instead — motion without a count we don't have.
 */
export default function UploadProgressBar({ progress }: { progress: UploadProgress }): React.ReactElement | null {
  if (progress.stage === "idle") return null;

  const isTransferring = progress.stage === "uploading" || progress.stage === "resuming";
  const fraction = uploadFraction(progress);
  const percent = Math.round(fraction * 100);

  return (
    <div
      style={{
        padding: "10px 16px",
        borderBottom: "1px solid rgba(255,255,255,0.07)",
        background: "rgba(255,255,255,0.04)",
        display: "flex",
        flexDirection: "column",
        gap: 6,
        flexShrink: 0,
      }}
      // Assistive tech gets the same two facts the sighted user does: what
      // stage this is, and how far along it is.
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={isTransferring ? percent : undefined}
      aria-valuetext={describeProgress(progress)}
      aria-busy
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <span style={{ fontSize: 12, color: ACCENT }}>{describeProgress(progress)}</span>
        {isTransferring && (
          <span style={{ fontSize: 12, fontVariantNumeric: "tabular-nums", color: "rgba(237,236,234,0.55)" }}>
            {percent}%
          </span>
        )}
      </div>

      <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.10)", overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            borderRadius: 2,
            background: ACCENT,
            width: isTransferring ? `${percent}%` : "100%",
            transition: "width 200ms linear",
            opacity: isTransferring ? 1 : 0.65,
            animation: isTransferring ? undefined : "cogneeUploadPulse 1.4s ease-in-out infinite",
          }}
        />
      </div>

      {isTransferring && progress.batchesTotal > 0 && (
        <span style={{ fontSize: 11, color: "rgba(237,236,234,0.4)" }}>
          Batch {Math.min(progress.batchesCompleted + 1, progress.batchesTotal)} of {progress.batchesTotal}
        </span>
      )}

      <style>{"@keyframes cogneeUploadPulse{0%{opacity:.35}50%{opacity:.8}100%{opacity:.35}}"}</style>
    </div>
  );
}

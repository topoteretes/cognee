"use client";

import { useCallback } from "react";

// Open-source stub — the real hook estimates upload cost against the
// workspace's remaining cloud credits (pdfjs/tiktoken-based estimator).
// There is no credit balance in local mode, so uploads are always allowed
// and the warning modal never renders. Exports mirror the SaaS module so
// shared UI (DatasetDetailPage, useBrainsData, LowBalanceWarningModal)
// compiles unchanged.

export interface RealEstimateProgress {
  processedFiles: number;
  totalFiles: number;
}

export type PendingLowBalanceWarning =
  | { kind: "point"; estimatedUsd: number; remainingUsd: number }
  | { kind: "range"; lowUsd: number; highUsd: number; remainingUsd: number };

export interface UseLowBalanceUploadWarningResult {
  pendingWarning: PendingLowBalanceWarning | null;
  isEstimating: boolean;
  estimateProgress: RealEstimateProgress | null;
  confirmUpload: (files: File[]) => Promise<boolean>;
  continueAnyway: () => void;
  cancel: () => void;
}

export function useLowBalanceUploadWarning(): UseLowBalanceUploadWarningResult {
  const confirmUpload = useCallback(async (): Promise<boolean> => true, []);
  const noop = useCallback((): void => undefined, []);

  return {
    pendingWarning: null,
    isEstimating: false,
    estimateProgress: null,
    confirmUpload,
    continueAnyway: noop,
    cancel: noop,
  };
}

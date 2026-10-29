import {
  DatasetProcessingTimeoutError,
  InsufficientCreditsProcessingError,
} from "@/modules/datasets/pollDatasetStatus";

export interface ProcessingErrorPresentation {
  title: string;
  message: string;
  isTimeout: boolean;
  isInsufficientCredits: boolean;
}

// Shown when LiteLLM's budget rejected the run mid-pipeline. Every parallel
// LLM call holds part of the budget while it runs, so a large file can hit the
// limit while the billing page still shows credit left — the copy says so,
// otherwise "out of credits" next to a positive balance reads as a bug.
export const INSUFFICIENT_CREDITS_PROCESSING_MESSAGE =
  "Files were added, but your workspace ran out of LLM budget while building the knowledge graph. " +
  "Large files are processed in many parallel steps, and each step holds part of your budget while it runs, " +
  "so the build can stop before your balance reaches $0. Add credits, then retry the build.";

// A DatasetProcessingTimeoutError only means the client gave up polling
// after pollDatasetStatus's deadline — the build may still be running
// server-side and complete later. The shared status poller
// (useDatasetStatuses) keeps checking independently of this one-off upload
// poll, so this must read as "still working", not "failed", and must not
// offer a build retry (which would kick off a redundant second build).
export function describeProcessingError(error: unknown): ProcessingErrorPresentation {
  if (error instanceof DatasetProcessingTimeoutError) {
    return {
      title: "Still building",
      message: "Files were added and the knowledge graph is still building — this can take longer than usual. The status will update automatically once it finishes.",
      isTimeout: true,
      isInsufficientCredits: false,
    };
  }
  if (error instanceof InsufficientCreditsProcessingError) {
    return {
      title: "Out of LLM budget",
      message: INSUFFICIENT_CREDITS_PROCESSING_MESSAGE,
      isTimeout: false,
      isInsufficientCredits: true,
    };
  }
  const errorMessage = error instanceof Error ? error.message : String(error);
  return {
    title: "Knowledge graph build failed",
    message: `Files were added, but building the knowledge graph failed: ${errorMessage}`,
    isTimeout: false,
    isInsufficientCredits: false,
  };
}

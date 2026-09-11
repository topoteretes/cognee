import { DatasetProcessingTimeoutError } from "@/modules/datasets/pollDatasetStatus";

export interface ProcessingErrorPresentation {
  title: string;
  message: string;
  isTimeout: boolean;
}

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
    };
  }
  const errorMessage = error instanceof Error ? error.message : String(error);
  return {
    title: "Knowledge graph build failed",
    message: `Files were added, but building the knowledge graph failed: ${errorMessage}`,
    isTimeout: false,
  };
}

import { IDLE_PROGRESS, type UploadProgress } from "@/modules/ingestion/uploadProgress";

export const mockUpload = jest.fn(() => Promise.resolve());

let progress: UploadProgress = IDLE_PROGRESS;
/** What the next render of the hook reports as the upload's progress. */
export function setMockProgress(next: UploadProgress): void {
  progress = next;
}

// A stand-in for tests of what hands files to the upload, not of the upload: the
// real hook needs a query client and a pod to talk to.
export function useBrainUpload(): {
  isUploading: boolean;
  stage: "idle";
  progress: UploadProgress;
  upload: typeof mockUpload;
  cancel: () => void;
} {
  return { isUploading: false, stage: "idle", progress, upload: mockUpload, cancel: () => undefined };
}

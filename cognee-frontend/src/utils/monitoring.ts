export function addBreadcrumb(_breadcrumb: {
  category?: string;
  level?: "info" | "warning" | "error";
  message?: string;
  data?: Record<string, unknown>;
}): void {}
export function captureMessage(
  _message: string,
  _context?: { level?: "info" | "warning" | "error"; tags?: Record<string, string>; extra?: Record<string, unknown> },
): void {}
export function captureException(_err: unknown, _context?: Record<string, unknown>): void {}
export function recordUploadSuccess(_durationMs: number, _totalBytes: number, _fileCount: number): void {}
export function recordUploadFailure(_errorName: string, _durationMs: number): void {}

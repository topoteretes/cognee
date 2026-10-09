export function captureException(_err: unknown, _context?: Record<string, unknown>): void {}
export function recordUploadSuccess(_durationMs: number, _totalBytes: number, _fileCount: number): void {}
export function recordUploadFailure(_errorName: string, _durationMs: number): void {}

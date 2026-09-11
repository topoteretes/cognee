import {
  describeProgress,
  formatBytes,
  IDLE_PROGRESS,
  uploadFraction,
  withEstimateStage,
  type UploadProgress,
} from "@/modules/ingestion/uploadProgress";

function progress(overrides: Partial<UploadProgress> = {}): UploadProgress {
  return { ...IDLE_PROGRESS, ...overrides };
}

describe("withEstimateStage", () => {
  it("shows the estimating stage while estimating and the upload leg is idle", () => {
    expect(withEstimateStage(IDLE_PROGRESS, true).stage).toBe("estimating");
  });

  it("is a no-op when not estimating", () => {
    expect(withEstimateStage(IDLE_PROGRESS, false)).toBe(IDLE_PROGRESS);
  });

  it("never overrides an upload that has already started", () => {
    const uploading = progress({ stage: "uploading", filesTotal: 10 });
    // Even if the estimate flag lingers, the real upload state must win.
    expect(withEstimateStage(uploading, true)).toBe(uploading);
  });
});

describe("describeProgress — estimating", () => {
  it("labels the estimate without a file counter", () => {
    expect(describeProgress(progress({ stage: "estimating" }))).toBe("Estimating cost…");
  });
});

describe("describeProgress — uploading", () => {
  // The label deliberately carries no "x of n" file counter. Files are
  // acknowledged a whole batch (10) at a time, so such a counter sat at 0/50
  // and then jumped to 10/50 — it read as a stuck upload. Bytes are the only
  // continuous signal available without per-file acknowledgement from the
  // backend, so they are what moves.
  it("states the selection size without pretending to per-file progress", () => {
    const line = describeProgress(
      progress({ stage: "uploading", filesTotal: 50, bytesTotal: 5_242_880, bytesSent: 1_048_576 }),
    );
    expect(line).toBe("Uploading 50 files — 1 MB of 5 MB");
    // Guards the regression: no "0/50", no "of 50 files".
    expect(line).not.toMatch(/\d+\s*\/\s*\d+/);
  });

  it("moves as bytes move, while the file count stays put", () => {
    const seen = [0, 1_048_576, 3_145_728, 5_242_880].map((bytesSent) =>
      describeProgress(progress({ stage: "uploading", filesTotal: 50, bytesTotal: 5_242_880, bytesSent })),
    );
    expect(new Set(seen).size).toBe(4);
    for (const line of seen) expect(line).toContain("Uploading 50 files");
  });

  it("says file, not files, for a single selection", () => {
    expect(
      describeProgress(progress({ stage: "uploading", filesTotal: 1, bytesTotal: 2048, bytesSent: 1024 })),
    ).toBe("Uploading 1 file — 1 KB of 2 KB");
  });

  it("is silent when idle", () => {
    expect(describeProgress(IDLE_PROGRESS)).toBe("");
  });
});

describe("describeProgress — resuming and processing", () => {
  // These DO carry counts, and legitimately: they report what the backend has
  // already accepted, not what is in flight.
  it("reports what already landed when resuming", () => {
    expect(
      describeProgress(progress({ stage: "resuming", filesTotal: 100, filesCompleted: 80 })),
    ).toBe("Resuming upload — 80/100 files already sent");
  });

  it("claims all files only when all of them got there", () => {
    expect(describeProgress(progress({ stage: "processing", filesTotal: 40, filesCompleted: 40 }))).toBe(
      "Building the knowledge graph — all 40 files uploaded",
    );
  });

  it("names what was lost when a degraded session could not send everything", () => {
    const line = describeProgress(
      progress({ stage: "processing", filesTotal: 100, filesCompleted: 30, unrecoverableFiles: 70 }),
    );
    expect(line).toContain("30 of 100");
    expect(line).toContain("70");
    expect(line).not.toContain("all 100");
  });
});

describe("uploadFraction", () => {
  it("uses bytes when they are known", () => {
    expect(uploadFraction(progress({ bytesTotal: 1000, bytesSent: 250 }))).toBe(0.25);
  });

  it("falls back to file counts when sizes are not known", () => {
    expect(uploadFraction(progress({ filesTotal: 4, filesCompleted: 1 }))).toBe(0.25);
  });

  it("never exceeds 1, even if the transport overshoots", () => {
    expect(uploadFraction(progress({ bytesTotal: 1000, bytesSent: 4000 }))).toBe(1);
  });

  it("is zero with nothing to measure", () => {
    expect(uploadFraction(IDLE_PROGRESS)).toBe(0);
  });
});

describe("formatBytes", () => {
  it("keeps small sizes in bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
  });

  it("steps up through the units", () => {
    expect(formatBytes(1024)).toBe("1 KB");
    expect(formatBytes(1024 * 1024)).toBe("1 MB");
    expect(formatBytes(1024 * 1024 * 1024)).toBe("1 GB");
  });

  it("keeps one decimal only where it carries information", () => {
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1024 * 15)).toBe("15 KB");
  });
});

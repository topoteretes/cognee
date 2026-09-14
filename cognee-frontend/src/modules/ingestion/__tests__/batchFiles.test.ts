import batchFiles, { totalBytes } from "@/modules/ingestion/batchFiles";

function file(name: string, size: number): File {
  const f = new File(["x"], name, { type: "text/plain" });
  Object.defineProperty(f, "size", { value: size });
  return f;
}

describe("batchFiles", () => {
  it("splits on the file-count bound", () => {
    const batches = batchFiles(
      Array.from({ length: 25 }, (_, i) => file(`f${i}.txt`, 10)),
      { maxFiles: 10, maxBytes: 1_000_000 },
    );
    expect(batches.map((b) => b.files.length)).toEqual([10, 10, 5]);
    expect(batches.map((b) => b.index)).toEqual([0, 1, 2]);
  });

  it("splits on the byte bound before the count bound", () => {
    const batches = batchFiles(
      [file("a", 60), file("b", 60), file("c", 20)],
      { maxFiles: 10, maxBytes: 100 },
    );
    expect(batches.map((b) => b.files.map((f) => f.name))).toEqual([["a"], ["b", "c"]]);
    expect(batches.map((b) => b.bytes)).toEqual([60, 80]);
  });

  it("gives a file larger than the byte bound its own batch rather than dropping it", () => {
    const batches = batchFiles([file("small", 10), file("huge", 999), file("after", 10)], {
      maxFiles: 10,
      maxBytes: 100,
    });
    expect(batches.map((b) => b.files.map((f) => f.name))).toEqual([["small"], ["huge"], ["after"]]);
    // Nothing is silently lost — the oversized file still goes up.
    expect(batches.flatMap((b) => b.files)).toHaveLength(3);
  });

  it("preserves selection order, so the pending remainder is always a suffix", () => {
    const files = Array.from({ length: 30 }, (_, i) => file(`f${i}.txt`, 10));
    const flattened = batchFiles(files, { maxFiles: 7, maxBytes: 1_000_000 }).flatMap((b) => b.files);
    expect(flattened.map((f) => f.name)).toEqual(files.map((f) => f.name));
  });

  it("returns no batches for an empty selection", () => {
    expect(batchFiles([])).toEqual([]);
  });

  it("sums bytes across the selection", () => {
    expect(totalBytes([file("a", 5), file("b", 7)])).toBe(12);
  });
});

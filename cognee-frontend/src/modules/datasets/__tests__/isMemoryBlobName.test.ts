import isMemoryBlobName from "@/modules/datasets/isMemoryBlobName";

describe("isMemoryBlobName", () => {
  it("returns true for a memory blob name with the .txt extension", () => {
    expect(isMemoryBlobName("text_0123456789abcdef.txt")).toBe(true);
  });

  it("returns true for a memory blob name without an extension", () => {
    expect(isMemoryBlobName("text_0123456789abcdef")).toBe(true);
  });

  it("returns true regardless of hash letter casing", () => {
    expect(isMemoryBlobName("text_0123456789ABCDEF.txt")).toBe(true);
  });

  it("returns false for a hash shorter than 16 hex characters", () => {
    expect(isMemoryBlobName("text_0123abc.txt")).toBe(false);
  });

  it("returns false for a normal uploaded filename", () => {
    expect(isMemoryBlobName("quarterly-report.pdf")).toBe(false);
  });

  it("returns false when the hash contains non-hex characters", () => {
    expect(isMemoryBlobName("text_0123456789zzzzzz.txt")).toBe(false);
  });

  it("returns false for a filename that merely starts with text_ but isn't a hash", () => {
    expect(isMemoryBlobName("text_summary.txt")).toBe(false);
  });
});

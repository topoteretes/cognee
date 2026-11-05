import { formatBytes } from "@/utils/formatBytes";

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

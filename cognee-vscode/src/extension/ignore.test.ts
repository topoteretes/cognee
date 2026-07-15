import { describe, expect, it } from "vitest";

import { buildIgnoreFilter } from "./ignore";

describe("buildIgnoreFilter", () => {
  it("returns undefined when there are no rules", () => {
    expect(buildIgnoreFilter(undefined, "")).toBeUndefined();
    expect(buildIgnoreFilter("   \n\n")).toBeUndefined();
  });

  it("ignores paths matched by .gitignore patterns", () => {
    const isIgnored = buildIgnoreFilter("*.log\ndist/\n")!;
    expect(isIgnored("app.log")).toBe(true);
    expect(isIgnored("dist/bundle.js")).toBe(true);
    expect(isIgnored("src/index.ts")).toBe(false);
  });

  it("combines .gitignore and .cogneeignore rules", () => {
    const isIgnored = buildIgnoreFilter("*.log", "secrets/\n")!;
    expect(isIgnored("a.log")).toBe(true);
    expect(isIgnored("secrets/key.txt")).toBe(true);
    expect(isIgnored("README.md")).toBe(false);
  });

  it("honors git negation (re-include) rules", () => {
    const isIgnored = buildIgnoreFilter("*.log\n!keep.log")!;
    expect(isIgnored("debug.log")).toBe(true);
    expect(isIgnored("keep.log")).toBe(false);
  });

  it("normalizes backslashes and never ignores a blank path", () => {
    const isIgnored = buildIgnoreFilter("dist/")!;
    expect(isIgnored("dist\\bundle.js")).toBe(true);
    expect(isIgnored("")).toBe(false);
  });
});

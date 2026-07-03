import { describe, expect, it } from "vitest";

import { basenameOf, knownPathMatches, normalizeRelativePath, stemOf } from "./paths";

describe("normalizeRelativePath", () => {
  it("uses forward slashes and strips ./ and leading/trailing slashes", () => {
    expect(normalizeRelativePath("./themes\\_welcome\\composer.json")).toBe(
      "themes/_welcome/composer.json",
    );
    expect(normalizeRelativePath("/a/b/")).toBe("a/b");
    expect(normalizeRelativePath("  x  ")).toBe("x");
  });
});

describe("basenameOf", () => {
  it("returns the final segment for posix and windows separators", () => {
    expect(basenameOf("themes/_welcome/composer.json")).toBe("composer.json");
    expect(basenameOf("a\\b\\c.ts")).toBe("c.ts");
    expect(basenameOf("composer.json")).toBe("composer.json");
  });
});

describe("stemOf", () => {
  it("drops only the final extension", () => {
    expect(stemOf("composer.json")).toBe("composer");
    expect(stemOf("themes/x/README.md")).toBe("README");
    expect(stemOf("archive.tar.gz")).toBe("archive.tar");
    expect(stemOf("Makefile")).toBe("Makefile");
  });
});

describe("knownPathMatches", () => {
  const candidates = ["app/README.md", "themes/_welcome/README.md", "README.md"];

  it("returns indices of candidates present in knownPaths (normalization-insensitive)", () => {
    expect(knownPathMatches(candidates, ["themes/_welcome/README.md"])).toEqual([1]);
    expect(knownPathMatches(candidates, ["./themes/_welcome/README.md"])).toEqual([1]);
    expect(knownPathMatches(candidates, ["app/README.md", "README.md"])).toEqual([0, 2]);
  });

  it("returns an empty list when nothing matches", () => {
    expect(knownPathMatches(candidates, [])).toEqual([]);
    expect(knownPathMatches(candidates, ["other/README.md"])).toEqual([]);
  });
});

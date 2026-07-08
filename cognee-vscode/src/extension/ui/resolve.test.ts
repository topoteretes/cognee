import { describe, expect, it } from "vitest";

import {
  contentMatchesSnippet,
  extractSourcePath,
  normalizeWhitespace,
  snippetSearchNeedles,
  stripProvenanceHeader,
} from "./resolve";

describe("extractSourcePath", () => {
  it("reads the path from a multi-line provenance header with a line range", () => {
    const snippet = "Source: themes/aurora/README.md (lines 1-5)\n\n# Aurora theme for Mautic";
    expect(extractSourcePath(snippet)).toBe("themes/aurora/README.md");
  });

  it("reads the path when the header is collapsed onto one line", () => {
    const snippet = "Source: themes/_welcome/README.md (lines 1-5) # Welcome theme for Mautic";
    expect(extractSourcePath(snippet)).toBe("themes/_welcome/README.md");
  });

  it("reads a whole-file header without a line range", () => {
    expect(extractSourcePath("Source: README.md\n\n# Root")).toBe("README.md");
  });

  it("returns undefined when there is no provenance header", () => {
    expect(extractSourcePath("# Aurora theme for Mautic")).toBeUndefined();
    expect(extractSourcePath(undefined)).toBeUndefined();
  });
});

describe("stripProvenanceHeader", () => {
  it("removes a multi-line header, leaving the file's real text", () => {
    const snippet = "Source: themes/aurora/README.md (lines 1-5)\n\n# Aurora theme for Mautic";
    expect(stripProvenanceHeader(snippet)).toBe("# Aurora theme for Mautic");
  });

  it("removes a collapsed single-line header", () => {
    const snippet = "Source: themes/aurora/README.md (lines 1-5) # Aurora theme for Mautic";
    expect(stripProvenanceHeader(snippet)).toBe("# Aurora theme for Mautic");
  });

  it("leaves a snippet without a header untouched", () => {
    expect(stripProvenanceHeader("Project note: rotate weekly.")).toBe("Project note: rotate weekly.");
  });
});

describe("normalizeWhitespace", () => {
  it("collapses runs of whitespace and trims", () => {
    expect(normalizeWhitespace("  a\n\t b   c  ")).toBe("a b c");
  });
});

describe("contentMatchesSnippet", () => {
  const fileA = "# Root README\n\nThis is the main project readme with setup steps.";
  const fileB = "# App mirror\n\nRead-only mirror of the app folder. Do not edit.";

  it("matches the file that contains the snippet", () => {
    const snippet = "the main project readme with setup steps";
    expect(contentMatchesSnippet(fileA, snippet)).toBe(true);
    expect(contentMatchesSnippet(fileB, snippet)).toBe(false);
  });

  it("tolerates whitespace differences between snippet and file", () => {
    expect(contentMatchesSnippet("alpha   beta\n\ngamma", "alpha beta gamma")).toBe(true);
  });

  it("ignores snippets that are too short to be distinctive", () => {
    expect(contentMatchesSnippet(fileA, "the")).toBe(false);
  });
});

describe("snippetSearchNeedles", () => {
  it("returns the leading slice then a substantial first line", () => {
    const needles = snippetSearchNeedles("Revenue grew twelve percent.\nMore detail here.");
    expect(needles.length).toBeGreaterThan(0);
    expect(needles[0]).toContain("Revenue grew twelve percent.");
  });

  it("drops candidates that are too short", () => {
    expect(snippetSearchNeedles("hi")).toEqual([]);
  });
});

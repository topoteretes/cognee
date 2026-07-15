import { describe, expect, it } from "vitest";

import {
  contentMatchesSnippet,
  extractSourcePath,
  findSnippetRange,
  normalizeWhitespace,
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

describe("findSnippetRange", () => {
  it("locates a whitespace-collapsed snippet inside real (multi-line, indented) source", () => {
    const doc = "export function foo() {\n  return bar;\n}\n";
    // The server emits the snippet with whitespace collapsed to single spaces.
    const range = findSnippetRange(doc, "export function foo() { return bar; }");
    expect(range).toBeDefined();
    expect(doc.slice(range![0], range![1])).toBe("export function foo() {\n  return bar;\n}");
  });

  it("tolerates differing whitespace/indentation between snippet and file", () => {
    const doc = "line one\n\n    indented    block\tend";
    const range = findSnippetRange(doc, "indented block end")!;
    expect(doc.slice(range[0], range[1])).toBe("indented    block\tend");
  });

  it("ignores the trailing ellipsis the server adds to truncated snippets", () => {
    const doc = "Project note: the deployment canary token for the checkout service is SMOKE-1.";
    const range = findSnippetRange(doc, "Project note: the deployment cana…")!;
    expect(doc.slice(range[0], range[1])).toBe("Project note: the deployment cana");
  });

  it("returns undefined when the snippet is absent or too short to match", () => {
    expect(findSnippetRange("some document text here", "nope not present anywhere")).toBeUndefined();
    expect(findSnippetRange("whatever", "hi")).toBeUndefined();
  });
});

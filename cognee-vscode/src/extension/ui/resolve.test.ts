import { describe, expect, it } from "vitest";

import { contentMatchesSnippet, normalizeWhitespace, snippetSearchNeedles } from "./resolve";

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

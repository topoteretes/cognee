import { describe, expect, it } from "vitest";

import type { Citation, RecallResponseItem } from "../../core";
import { dedupeByFile, rankCitations, renderRecall } from "./render";

function citation(documentName: string, snippet: string): Citation {
  return { chunkLabel: "1", documentName, snippet, raw: `${documentName}: ${snippet}` };
}

describe("rankCitations", () => {
  const rootReadme = citation(
    "README",
    "Source: README.md (lines 1-409) Mautic Open Source Marketing Automation",
  );
  const auroraReadme = citation(
    "README",
    "Source: themes/aurora/README.md (lines 1-5) # Aurora theme for Mautic. This theme is managed centrally.",
  );

  it("ranks the query-relevant source above an incidental one", () => {
    const ranked = rankCitations([rootReadme, auroraReadme], "how to manage aurora theme");
    expect(ranked[0]).toBe(auroraReadme);
    expect(ranked[1]).toBe(rootReadme);
  });

  it("keeps Cognee's original order when scores tie (stable)", () => {
    const ranked = rankCitations([rootReadme, auroraReadme], "unrelated zzz query");
    expect(ranked[0]).toBe(rootReadme);
    expect(ranked[1]).toBe(auroraReadme);
  });

  it("is a no-op for a single citation", () => {
    expect(rankCitations([auroraReadme], "aurora")).toEqual([auroraReadme]);
  });
});

describe("dedupeByFile", () => {
  it("collapses multiple chunks of the same file into one entry, keeping the first", () => {
    const rootChunk1 = citation("README", "Source: README.md (lines 1-409) logo and intro");
    const rootChunk2 = citation("README", "Source: README.md (lines 1-409) contributors list");
    const aurora = citation("README", "Source: themes/aurora/README.md (lines 1-5) # Aurora theme");

    const unique = dedupeByFile([rootChunk1, rootChunk2, aurora]);
    expect(unique).toEqual([rootChunk1, aurora]);
  });

  it("falls back to the document name when there is no Source header", () => {
    const a = citation("a.ts", "some code");
    const aAgain = citation("a.ts", "more code from the same file");
    const b = citation("b.ts", "other code");
    expect(dedupeByFile([a, aAgain, b])).toEqual([a, b]);
  });
});

describe("renderRecall", () => {
  it("extracts the graph answer and parses its citations", () => {
    const items: RecallResponseItem[] = [
      {
        source: "graph",
        kind: "graph_completion",
        search_type: "GRAPH_COMPLETION",
        text: "The answer.\n\nEvidence:\n- chunk 1 of document a.ts (chunk_id: c1): snippet text",
      },
    ];

    const rendered = renderRecall(items);
    expect(rendered.isEmpty).toBe(false);
    expect(rendered.answer).toBe("The answer.");
    expect(rendered.citations).toHaveLength(1);
    expect(rendered.citations[0].documentName).toBe("a.ts");
  });

  it("uses `content` for context-source items", () => {
    const items: RecallResponseItem[] = [{ source: "graph_context", content: "context summary" }];
    expect(renderRecall(items).answer).toBe("context summary");
  });

  it("de-duplicates identical citations across items", () => {
    const evidence = "A.\n\nEvidence:\n- chunk 1 of document a.ts (chunk_id: c1): s";
    const items: RecallResponseItem[] = [
      { source: "graph", kind: "graph_completion", search_type: "GRAPH_COMPLETION", text: evidence },
      { source: "graph", kind: "graph_completion", search_type: "GRAPH_COMPLETION", text: evidence },
    ];
    expect(renderRecall(items).citations).toHaveLength(1);
  });

  it("reports empty when nothing is renderable", () => {
    expect(renderRecall([]).isEmpty).toBe(true);
  });
});

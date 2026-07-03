import { describe, expect, it } from "vitest";

import type { RecallResponseItem } from "../../core";
import { renderRecall } from "./render";

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

import { describe, expect, it } from "vitest";

import { parseAnswer } from "./evidence";

const WITH_EVIDENCE =
  "Revenue grew 12 percent.\n\nEvidence:\n" +
  "- chunk 3 of document report.pdf (chunk_id: chunk-1): Revenue grew 12 percent year over year.\n" +
  "- chunk 5 of document notes.md (chunk_id: chunk-2): More detail here.";

describe("parseAnswer", () => {
  it("returns the answer verbatim when there is no Evidence block", () => {
    const result = parseAnswer("Just a plain answer.");
    expect(result.answer).toBe("Just a plain answer.");
    expect(result.citations).toHaveLength(0);
  });

  it("splits the answer from the Evidence block and parses each citation", () => {
    const result = parseAnswer(WITH_EVIDENCE);
    expect(result.answer).toBe("Revenue grew 12 percent.");
    expect(result.citations).toHaveLength(2);
    expect(result.citations[0]).toMatchObject({
      chunkLabel: "3",
      documentName: "report.pdf",
      chunkId: "chunk-1",
      snippet: "Revenue grew 12 percent year over year.",
    });
    expect(result.citations[1]).toMatchObject({
      chunkLabel: "5",
      documentName: "notes.md",
      chunkId: "chunk-2",
    });
  });

  it("ignores malformed lines inside the Evidence block", () => {
    const result = parseAnswer("Answer.\n\nEvidence:\n- not a citation line\nrandom text");
    expect(result.answer).toBe("Answer.");
    expect(result.citations).toHaveLength(0);
  });
});

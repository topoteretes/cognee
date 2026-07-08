import { describe, expect, it } from "vitest";

import { parseAnswer } from "./evidence";

const WITH_EVIDENCE =
  "Revenue grew 12 percent.\n\nEvidence:\n" +
  "- chunk 3 of document report.pdf (chunk_id: chunk-1): Revenue grew 12 percent year over year.\n" +
  "- chunk 5 of document notes.md (chunk_id: chunk-2): More detail here.";

// Verbatim shape returned by Cognee Cloud today: the parenthetical leads with
// `data_id:` before `chunk_id:`, and the snippet is wrapped in double quotes.
const CLOUD_EVIDENCE =
  "The checkout canary is rotated every Friday.\n\nEvidence:\n" +
  "- chunk 1 of document smoke (data_id: 9b6475c0-5e22-52f9-a629-46159aeb60a9, " +
  'chunk_id: 5fecd7fc-51c1-52a0-9b47-74ff9024bd09): "Project note: the deployment ' +
  'canary token for the checkout service is SMOKE-1. It is rotated every Friday."';

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

  it("parses the real Cognee Cloud format (data_id before chunk_id, quoted snippet)", () => {
    const result = parseAnswer(CLOUD_EVIDENCE);
    expect(result.answer).toBe("The checkout canary is rotated every Friday.");
    expect(result.citations).toHaveLength(1);
    expect(result.citations[0]).toMatchObject({
      chunkLabel: "1",
      documentName: "smoke",
      dataId: "9b6475c0-5e22-52f9-a629-46159aeb60a9",
      chunkId: "5fecd7fc-51c1-52a0-9b47-74ff9024bd09",
    });
    // Surrounding quotes are stripped so snippet matching works against file text.
    expect(result.citations[0].snippet).toBe(
      "Project note: the deployment canary token for the checkout service is " +
        "SMOKE-1. It is rotated every Friday.",
    );
  });

  it("ignores malformed lines inside the Evidence block", () => {
    const result = parseAnswer("Answer.\n\nEvidence:\n- not a citation line\nrandom text");
    expect(result.answer).toBe("Answer.");
    expect(result.citations).toHaveLength(0);
  });
});

import describeTextBlob, {
  blobsToRead,
  clipTitle,
  connectorRepository,
  isTextBlobName,
  needsContentRead,
  textBlobLabel,
} from "../describeTextBlob";

describe("isTextBlobName", () => {
  it("matches a hashed text blob, with or without the extension", () => {
    expect(isTextBlobName("text_0123456789abcdef.txt")).toBe(true);
    expect(isTextBlobName("text_0123456789abcdef")).toBe(true);
    expect(isTextBlobName("text_0123456789ABCDEF.txt")).toBe(true);
  });

  it("does not match a hash too short to be one, or a real filename", () => {
    expect(isTextBlobName("text_0123abc.txt")).toBe(false);
    expect(isTextBlobName("text_0123456789zzzzzz.txt")).toBe(false);
    expect(isTextBlobName("text_summary.txt")).toBe(false);
    expect(isTextBlobName("roadmap.docx")).toBe(false);
  });
});

describe("describeTextBlob", () => {
  it("reads the session id a memory blob carries", () => {
    expect(describeTextBlob("Session ID: abc-123\nrest of it")).toEqual({
      kind: "memory",
      sessionId: "abc-123",
    });
  });

  it("titles anything else from its first meaningful line", () => {
    expect(describeTextBlob("\n\n# Q3 roadmap\n\nbody")).toEqual({ kind: "text", title: "Q3 roadmap" });
  });

  it("truncates a long opening line instead of running it across the row", () => {
    const description = describeTextBlob("x".repeat(200));
    expect(description).toEqual({ kind: "text", title: `${"x".repeat(80)}…` });
  });

  it("gives no title to a blob with nothing in it", () => {
    expect(describeTextBlob("   \n\n")).toEqual({ kind: "text", title: "" });
  });

  // A connector's content is not something the user typed into memory, and
  // labelling it "Memory" claims an origin nobody checked.
  it("does not call a blob without a session header a memory", () => {
    expect(describeTextBlob("# README\n\nCognee reads this repo.").kind).toBe("text");
  });
});

describe("textBlobLabel", () => {
  it("names a memory by its session", () => {
    expect(textBlobLabel({ kind: "memory", sessionId: "abc-123" })).toBe("Memory · abc-123");
  });

  it("falls back to 'Text' while unread, on a failed read, and with no title", () => {
    expect(textBlobLabel(undefined)).toBe("Text");
    expect(textBlobLabel(null)).toBe("Text");
    expect(textBlobLabel({ kind: "text", title: "" })).toBe("Text");
  });

  it("uses the derived title when there is one", () => {
    expect(textBlobLabel({ kind: "text", title: "Q3 roadmap" })).toBe("Q3 roadmap");
  });
});

const HASHED = "text_0123456789abcdef0123456789abcdef";
const REPO_LABEL = "code_repo:github.com-topoteretes-cognee-website";

describe("connectorRepository", () => {
  it("reads the repository off a connector-synced row's label", () => {
    expect(connectorRepository({ name: HASHED, label: REPO_LABEL })).toBe(
      "github.com-topoteretes-cognee-website",
    );
  });

  it("is null for any other label, a missing one, or a prefix with nothing after it", () => {
    for (const label of [undefined, null, 42, "", "upload:roadmap", "code_repo:", "code_repo:   "]) {
      expect(connectorRepository({ name: HASHED, label })).toBeNull();
    }
  });

  it("never renames a row that already has a real filename, whatever its label says", () => {
    expect(connectorRepository({ name: "roadmap.docx", label: REPO_LABEL })).toBeNull();
  });

  it("returns a long label whole, so two long repositories stay told apart", () => {
    const prefix = "a".repeat(80);

    expect(connectorRepository({ name: HASHED, label: `code_repo:${prefix}-one` })).toBe(`${prefix}-one`);
    expect(connectorRepository({ name: HASHED, label: `code_repo:${prefix}-two` })).toBe(`${prefix}-two`);
  });
});

describe("clipTitle", () => {
  it("leaves a title that fits alone and clips one that does not", () => {
    expect(clipTitle("short")).toBe("short");
    expect(clipTitle("a".repeat(80))).toBe("a".repeat(80));
    expect(clipTitle("a".repeat(300))).toBe(`${"a".repeat(80)}…`);
  });
});

describe("needsContentRead", () => {
  it("is true for an anonymous hashed blob", () => {
    expect(needsContentRead({ name: HASHED })).toBe(true);
    expect(needsContentRead({ name: HASHED, label: "upload:notes" })).toBe(true);
  });

  it("is false once the label already names a repository, so no read is spent on it", () => {
    expect(needsContentRead({ name: HASHED, label: REPO_LABEL })).toBe(false);
  });

  it("is false for a file with a real name", () => {
    expect(needsContentRead({ name: "roadmap.docx" })).toBe(false);
  });
});

describe("blobsToRead", () => {
  const anonymous = { id: "a", name: HASHED };
  const repository = { id: "r", name: HASHED, label: REPO_LABEL };
  const named = { id: "n", name: "roadmap.docx" };
  const read = { id: "d", name: HASHED };

  it("selects only unnamed hashed blobs, so a synced repository costs no request", () => {
    expect(blobsToRead([anonymous, repository, named], {})).toEqual([anonymous]);
  });

  it("skips a blob whose read has already answered, even if that read failed", () => {
    expect(blobsToRead([anonymous, read], { d: null })).toEqual([anonymous]);
  });
});

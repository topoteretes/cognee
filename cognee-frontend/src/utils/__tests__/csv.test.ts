import { downloadCsv, escapeCsvCell, toCsv } from "@/utils/csv";

describe("escapeCsvCell", () => {
  it("quotes every cell and doubles an embedded quote", () => {
    expect(escapeCsvCell("plain")).toBe('"plain"');
    expect(escapeCsvCell('the "main" set')).toBe('"the ""main"" set"');
  });

  it("keeps a comma or newline inside one cell", () => {
    expect(escapeCsvCell("sales, EMEA")).toBe('"sales, EMEA"');
    expect(escapeCsvCell("two\nlines")).toBe('"two\nlines"');
  });

  it("defuses a leading formula character, which quoting alone does not", () => {
    // RFC 4180 quotes are stripped at parse and the spreadsheet then evaluates
    // the cell. CWE-1236. The values here are caller-supplied: dataset names,
    // agent names, and session ids the pod validates only as non-empty.
    for (const lead of ["=", "+", "-", "@", "\t", "\r", "\n"]) {
      const out = escapeCsvCell(`${lead}cmd|' /C calc'!A0`);
      expect(out.startsWith(`"'${lead}`)).toBe(true);
    }
  });

  it("defuses a payload that leads with a newline rather than the sigil itself", () => {
    // A newline is not in FORMULA_LEAD's usual company (=, +, -, @) but reaches
    // this function the same way they do: CSV quoting is what lets a cell hold
    // a literal newline at all, and a spreadsheet still evaluates what follows
    // it as a formula. Missing from the set left this class of payload through
    // undefused (CWE-1236) and invisible to the loop above, which iterated the
    // same incomplete set this fixes.
    const out = escapeCsvCell("\n=cmd|' /C calc'!A0");
    expect(out.startsWith(`"'\n`)).toBe(true);
  });

  it("defuses the real-world payloads, not just the sigil", () => {
    expect(escapeCsvCell('=HYPERLINK("http://evil","click")'))
      .toBe(`"'=HYPERLINK(""http://evil"",""click"")"`);
    expect(escapeCsvCell("=cmd|' /C calc'!A0")).toBe(`"'=cmd|' /C calc'!A0"`);
  });

  it("leaves an ordinary value untouched", () => {
    expect(escapeCsvCell("5")).toBe('"5"');
    expect(escapeCsvCell("my_docs")).toBe('"my_docs"');
    expect(escapeCsvCell("")).toBe('""');
    expect(escapeCsvCell("2026-09-07T10:00:00")).toBe('"2026-09-07T10:00:00"');
  });

  it("prefixes a bare negative number, which is the accepted cost of the defusal", () => {
    // A spreadsheet reads a leading "-" as a formula, so it has to be defused
    // like the others. Verified separately that nothing in either export writes
    // a negative: activity tokens are sums of non-negative ints, cost derives
    // from those, duration is clamped at zero, and memory-gap's formatScore is
    // toFixed(1) over a non-negative judge score.
    expect(escapeCsvCell("-5")).toBe(`"'-5"`);
  });

  it("survives a non-string, because these cells come from unvalidated wire data", () => {
    // It threw inside a click handler, where React 19 sends the error to the
    // global handler and the button appears to do nothing.
    expect(escapeCsvCell(undefined as unknown as string)).toBe('""');
    expect(escapeCsvCell(null as unknown as string)).toBe('""');
    expect(escapeCsvCell(5 as unknown as string)).toBe('"5"');
  });
});

describe("toCsv", () => {
  it("writes a header row and one line per row", () => {
    expect(toCsv(["A", "B"], [["1", "2"], ["3", "4"]]))
      .toBe('"A","B"\n"1","2"\n"3","4"');
  });

  it("writes only the header when there are no rows", () => {
    expect(toCsv(["A"], [])).toBe('"A"');
  });

  it("defuses a formula in a body cell", () => {
    expect(toCsv(["Dataset"], [["=1+1"]])).toBe('"Dataset"\n"\'=1+1"');
  });
});

describe("downloadCsv", () => {
  const originalCreate = URL.createObjectURL;
  const originalRevoke = URL.revokeObjectURL;

  afterEach(() => {
    Object.assign(URL, { createObjectURL: originalCreate, revokeObjectURL: originalRevoke });
    jest.restoreAllMocks();
  });

  it("reports success rather than returning nothing, so a caller can react", () => {
    Object.assign(URL, { createObjectURL: () => "blob:x", revokeObjectURL: () => {} });
    expect(downloadCsv("f.csv", '"A"')).toBe(true);
  });

  it("reports failure instead of throwing into a click handler", () => {
    // A throw inside onClick reaches React 19's global error handler, not an
    // error boundary, so the button would appear to do nothing at all.
    Object.assign(URL, {
      createObjectURL: () => { throw new Error("no blob"); },
      revokeObjectURL: () => {},
    });
    const warn = jest.spyOn(console, "error").mockImplementation(() => {});

    expect(downloadCsv("f.csv", '"A"')).toBe(false);
    expect(warn).toHaveBeenCalled();
  });

  it("attaches the anchor and defers the revoke, which some browsers need", () => {
    let revoked = false;
    Object.assign(URL, { createObjectURL: () => "blob:x", revokeObjectURL: () => { revoked = true; } });
    jest.useFakeTimers();
    try {
      expect(downloadCsv("f.csv", '"A"')).toBe(true);
      // Not yet: revoking in the same task as the click cancels the download in
      // some browsers.
      expect(revoked).toBe(false);
      jest.runOnlyPendingTimers();
      expect(revoked).toBe(true);
      // And the anchor is cleaned up rather than left in the document.
      expect(document.querySelectorAll("a[download]")).toHaveLength(0);
    } finally {
      jest.useRealTimers();
    }
  });
});

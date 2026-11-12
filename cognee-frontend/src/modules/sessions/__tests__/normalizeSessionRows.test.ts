import { normalizeSessionRows, sumTokens } from "@/modules/sessions/getSessions";

describe("sumTokens", () => {
  it("adds two measured sides", () => {
    expect(sumTokens(10, 5)).toBe(15);
  });

  it("reports a measured zero as zero", () => {
    expect(sumTokens(0, 0)).toBe(0);
  });

  it("reports nothing when neither side was measured", () => {
    expect(sumTokens(null, null)).toBeNull();
  });

  it("reports nothing when only one side was measured", () => {
    // A total that omits one half is not a total. The case that matters is
    // `0 + null`, which used to render as a confident 0 and read exactly like a
    // row that genuinely spent nothing.
    expect(sumTokens(40, null)).toBeNull();
    expect(sumTokens(null, 40)).toBeNull();
    expect(sumTokens(0, null)).toBeNull();
  });
});

describe("normalizeSessionRows", () => {
  it("keeps a measured zero and a real count apart from an absent column", () => {
    const [zero, real, absent] = normalizeSessionRows([
      { session_id: "a", user_id: "u", tokens_in: 0, tokens_out: 0 },
      { session_id: "b", user_id: "u", tokens_in: 10, tokens_out: 5 },
      { session_id: "c", user_id: "u" },
    ]);

    expect([zero.tokens_in, zero.tokens_out]).toEqual([0, 0]);
    expect([real.tokens_in, real.tokens_out]).toEqual([10, 5]);
    expect([absent.tokens_in, absent.tokens_out]).toEqual([null, null]);
  });

  it("refuses a token count that is not a number", () => {
    // `isSessionsPage` proves only that `sessions` is an array, so these fields
    // arrived unchecked behind a non-nullable type. A string survived `|| 0`
    // into `+` as concatenation, which is the bug the activity endpoint's own
    // normalizer was written to close.
    const [row] = normalizeSessionRows([
      { session_id: "a", user_id: "u", tokens_in: "500", tokens_out: 5 },
    ]);

    expect(row.tokens_in).toBeNull();
    expect(sumTokens(row.tokens_in, row.tokens_out)).toBeNull();
  });

  it("refuses a token count that is not finite", () => {
    const [row] = normalizeSessionRows([
      { session_id: "a", user_id: "u", tokens_in: Number.NaN, tokens_out: Number.POSITIVE_INFINITY },
    ]);

    expect([row.tokens_in, row.tokens_out]).toEqual([null, null]);
  });

  it("counts no errors rather than reporting an unknown error count", () => {
    // `error_count` drives a red badge, and null would render as "null errors".
    // Zero is the right default here precisely because the column is a count of
    // things that happened, not a measurement of something that may be absent.
    const [row] = normalizeSessionRows([{ session_id: "a", user_id: "u" }]);

    expect(row.error_count).toBe(0);
  });

  it("drops an entry that is not an object", () => {
    expect(normalizeSessionRows([null, "session", 7, { session_id: "a", user_id: "u" }])).toHaveLength(1);
  });

  it("gives a row with no identifiers empty strings rather than undefined", () => {
    const [row] = normalizeSessionRows([{ tokens_in: 1, tokens_out: 1 }]);

    expect(row.session_id).toBe("");
    expect(row.user_id).toBe("");
  });
});

import { DEFAULT_SEARCH_TYPE, isRepositoryDataset, isSearchTypeId, SEARCH_TYPES, searchTypeOption } from "../searchTypes";

describe("searchTypes", () => {
  it("starts on Text, which is what every search was before the selector existed", () => {
    expect(DEFAULT_SEARCH_TYPE).toBe("text");
  });

  it("offers exactly the two types the page renders", () => {
    expect(SEARCH_TYPES.map((t) => t.id)).toEqual(["text", "code"]);
    expect(SEARCH_TYPES.map((t) => t.label)).toEqual(["Text", "Code"]);
  });

  // Code is a name lookup, not a question, and the box is the same box — the
  // placeholder is the only thing that says so before the first search.
  it("tells the two apart by what to type", () => {
    expect(searchTypeOption("text").placeholder).toMatch(/question/i);
    expect(searchTypeOption("code").placeholder).toMatch(/function, class or file name/i);
  });

  it("gives each type starting points of its own", () => {
    expect(searchTypeOption("code").suggestions).toContain("Show the architecture");
    expect(searchTypeOption("text").suggestions).not.toContain("Show the architecture");
  });

  it.each(["text", "code"])("recognises %s as a stored id", (id) => {
    expect(isSearchTypeId(id)).toBe(true);
  });

  // The key is per pod and long-lived, so it outlives any rename of these ids.
  // "toString" and "constructor" are on every object's prototype; a lookup
  // that used `in` would have accepted them.
  it.each([null, "", "TEXT", "graph", "CODE", "HYBRID_COMPLETION", "toString", "constructor"])(
    "rejects %p, which a past build or a hand edit could have left behind",
    (stored) => {
      expect(isSearchTypeId(stored)).toBe(false);
    },
  );

  describe("isRepositoryDataset", () => {
    // Cloud connectors name datasets {provider}_{slug}; cognee's own GitHub
    // sync uses the same prefix.
    it.each(["github_123456", "github_acct_42", "github_topoteretes"])("recognises %s", (name) => {
      expect(isRepositoryDataset(name)).toBe(true);
    });

    it.each(["main_dataset", "slack_t01abcd", "my_github_notes", "GitHub_repo", "github"])("does not claim %s", (name) => {
      expect(isRepositoryDataset(name)).toBe(false);
    });
  });
});

import { cutToLength } from "../cutToLength";

describe("cutToLength", () => {
  it("leaves a value that fits as it is", () => {
    expect(cutToLength("abc", 3)).toBe("abc");
  });

  it("cuts a longer value to the limit", () => {
    expect(cutToLength("abcdef", 4)).toBe("abcd");
  });

  // 😀 is two UTF-16 units; cutting between them leaves a lone surrogate.
  it("drops a character it would have cut in half", () => {
    const cut = cutToLength(`${"a".repeat(127)}😀b`, 128);

    expect(cut).toBe("a".repeat(127));
    expect(cut.isWellFormed()).toBe(true);
    expect(() => encodeURIComponent(cut)).not.toThrow();
  });

  it("keeps a whole character that ends exactly at the limit", () => {
    expect(cutToLength(`${"a".repeat(126)}😀b`, 128)).toBe(`${"a".repeat(126)}😀`);
  });
});

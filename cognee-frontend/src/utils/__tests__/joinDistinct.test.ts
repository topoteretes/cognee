import { joinDistinct } from "../joinDistinct";

describe("joinDistinct", () => {
  it("lists each value once, in the order first seen", () => {
    expect(joinDistinct(["b", "a", "b", "c", "a"])).toBe("b,a,c");
  });

  it("is empty for no values", () => {
    expect(joinDistinct([])).toBe("");
  });
});

import { readFileSync } from "fs";
import { join } from "path";

const globals = readFileSync(join(__dirname, "../../../../app/globals.css"), "utf8");

describe("the brain's breathing", () => {
  it("is switched off for reduced motion, leaving a static brain", () => {
    const reduced = globals.match(
      /@media \(prefers-reduced-motion: reduce\)\s*\{\s*\.brain-cell-pulse\s*\{\s*animation:\s*none;/,
    );

    expect(reduced).not.toBeNull();
  });

  it.each(["t0", "t1", "t2", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"])(
    "has the pulse-%s class the squares' beats use",
    (beat) => {
      expect(globals).toContain(`.pulse-${beat} {`);
    },
  );
});

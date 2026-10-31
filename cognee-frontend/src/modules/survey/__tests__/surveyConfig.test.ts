import { scoreBucketFor } from "../surveyConfig";

describe("scoreBucketFor", () => {
  it("classifies 0 as a detractor", () => {
    expect(scoreBucketFor(0)).toBe("detractor");
  });

  it("classifies the detractor boundary (6) as a detractor", () => {
    expect(scoreBucketFor(6)).toBe("detractor");
  });

  it("classifies the passive boundary (7) as passive", () => {
    expect(scoreBucketFor(7)).toBe("passive");
  });

  it("classifies the passive boundary (8) as passive", () => {
    expect(scoreBucketFor(8)).toBe("passive");
  });

  it("classifies the promoter boundary (9) as a promoter", () => {
    expect(scoreBucketFor(9)).toBe("promoter");
  });

  it("classifies 10 as a promoter", () => {
    expect(scoreBucketFor(10)).toBe("promoter");
  });
});

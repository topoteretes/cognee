import { HttpError } from "@/services/http/errors";
import { isInsufficientCreditsError, parseInsufficientCreditsOperation } from "@/utils/insufficientCredits";

describe("isInsufficientCreditsError", () => {
  it("returns true for an HttpError with status 402", () => {
    expect(isInsufficientCreditsError(new HttpError(402, "Payment Required", "no credits"))).toBe(true);
  });

  it("returns false for an HttpError with a different status", () => {
    expect(isInsufficientCreditsError(new HttpError(500, "Internal Server Error", "oops"))).toBe(false);
  });

  it("returns false for a non-HttpError value", () => {
    expect(isInsufficientCreditsError(new Error("plain error"))).toBe(false);
  });
});

describe("parseInsufficientCreditsOperation", () => {
  it("reads the operation directly from a structured body (CLO-305 pod)", () => {
    const error = new HttpError(402, "Payment Required", "Insufficient credits to run cognify.", {
      detail: "Insufficient credits to run cognify.",
      reason: "insufficient_credits",
      operation: "cognify",
      remaining_usd: 0.32,
    });

    expect(parseInsufficientCreditsOperation(error)).toBe("cognify");
  });

  it("falls back to regex-parsing the free-text message when the body has no operation field (pre-CLO-305 pod)", () => {
    const error = new HttpError(
      402,
      "Payment Required",
      "Insufficient credits to run search. Only $0.10 of credits remain.",
      "Insufficient credits to run search. Only $0.10 of credits remain.",
    );

    expect(parseInsufficientCreditsOperation(error)).toBe("search");
  });

  it("returns null when neither the body nor the message contain an operation", () => {
    const error = new HttpError(402, "Payment Required", "Something went wrong.");

    expect(parseInsufficientCreditsOperation(error)).toBeNull();
  });

  it("prefers the structured field even if the free-text message would parse differently", () => {
    const error = new HttpError(402, "Payment Required", "Insufficient credits to run remember.", {
      detail: "Insufficient credits to run remember.",
      operation: "cognify",
    });

    expect(parseInsufficientCreditsOperation(error)).toBe("cognify");
  });
});

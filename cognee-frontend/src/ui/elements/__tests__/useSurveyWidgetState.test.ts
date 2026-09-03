const mockSubmitSurveyResponse = jest.fn();
jest.mock("@/modules/survey/submitSurveyResponse", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockSubmitSurveyResponse(...args),
}));

jest.mock("@/modules/analytics", () => ({
  trackEvent: jest.fn(),
}));

import { act, renderHook } from "@testing-library/react";
import { useSurveyWidgetState } from "../useSurveyWidgetState";

const RESPONSE_ID = "response-123";

describe("useSurveyWidgetState", () => {
  afterEach(() => {
    mockSubmitSurveyResponse.mockReset();
  });

  it("starts on the score step with no score selected", () => {
    const { result } = renderHook(() => useSurveyWidgetState(RESPONSE_ID, jest.fn()));

    expect(result.current.step).toBe("score");
    expect(result.current.score).toBeNull();
    expect(result.current.bucket).toBeNull();
  });

  it("advances to the followup step with the matching bucket after selecting a score", () => {
    const { result } = renderHook(() => useSurveyWidgetState(RESPONSE_ID, jest.fn()));

    act(() => result.current.selectScore(9));

    expect(result.current.step).toBe("followup");
    expect(result.current.score).toBe(9);
    expect(result.current.bucket).toBe("promoter");
  });

  it("clears a prior answer and consent when the user goes back and picks a different score", () => {
    const { result } = renderHook(() => useSurveyWidgetState(RESPONSE_ID, jest.fn()));

    act(() => result.current.selectScore(10));
    act(() => result.current.setAnswer("I love the recall quality."));
    act(() => result.current.setConsentToQuote(true));
    act(() => result.current.goBackToScore());
    act(() => result.current.selectScore(5));

    expect(result.current.answer).toBe("");
    expect(result.current.consentToQuote).toBe(false);
    expect(result.current.bucket).toBe("detractor");
  });

  it("submits the promoter's consent flag when the user actually checked it", async () => {
    mockSubmitSurveyResponse.mockResolvedValue({ id: RESPONSE_ID, score: 10, scoreBucket: "promoter" });
    const { result } = renderHook(() => useSurveyWidgetState(RESPONSE_ID, jest.fn()));

    act(() => result.current.selectScore(10));
    act(() => result.current.setConsentToQuote(true));
    await act(async () => result.current.send());

    expect(mockSubmitSurveyResponse).toHaveBeenCalledWith(
      expect.objectContaining({ responseId: RESPONSE_ID, score: 10, consentToQuote: true, followupQuestionId: "what_valued" }),
    );
  });

  it("never submits consent for a bucket whose consent checkbox is never shown", async () => {
    mockSubmitSurveyResponse.mockResolvedValue({ id: RESPONSE_ID, score: 3, scoreBucket: "detractor" });
    const { result } = renderHook(() => useSurveyWidgetState(RESPONSE_ID, jest.fn()));

    // Detractor's UI never renders a consent checkbox, so setConsentToQuote
    // should be unreachable in practice — this asserts the defensive gate in
    // `submit` still holds even if some other path set it anyway.
    act(() => result.current.selectScore(3));
    act(() => result.current.setConsentToQuote(true));
    await act(async () => result.current.send());

    expect(mockSubmitSurveyResponse).toHaveBeenCalledWith(expect.objectContaining({ consentToQuote: false }));
  });

  it("submits null follow-up fields and moves to thanks when skipped", async () => {
    mockSubmitSurveyResponse.mockResolvedValue({ id: RESPONSE_ID, score: 2, scoreBucket: "detractor" });
    const { result } = renderHook(() => useSurveyWidgetState(RESPONSE_ID, jest.fn()));

    act(() => result.current.selectScore(2));
    act(() => result.current.setAnswer("some typed text"));
    await act(async () => result.current.skip());

    expect(mockSubmitSurveyResponse).toHaveBeenCalledWith(
      expect.objectContaining({ followupQuestionId: null, followupAnswer: null, consentToQuote: false }),
    );
    expect(result.current.step).toBe("thanks");
  });

  it("shows an error and stays on the followup step when submission fails", async () => {
    mockSubmitSurveyResponse.mockRejectedValue(new Error("network down"));
    const errorSpy = jest.spyOn(console, "error").mockImplementation(() => undefined);
    const { result } = renderHook(() => useSurveyWidgetState(RESPONSE_ID, jest.fn()));

    act(() => result.current.selectScore(7));
    await act(async () => result.current.send());

    expect(result.current.step).toBe("followup");
    expect(result.current.error).toBe("Could not send your answer. Please try again.");
    expect(result.current.sending).toBe(false);
    errorSpy.mockRestore();
  });

  it("calls onDone when closed", () => {
    const onDone = jest.fn();
    const { result } = renderHook(() => useSurveyWidgetState(RESPONSE_ID, onDone));

    act(() => result.current.close());

    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("auto-closes a few seconds after reaching the thanks step", async () => {
    jest.useFakeTimers();
    const onDone = jest.fn();
    mockSubmitSurveyResponse.mockResolvedValue({ id: RESPONSE_ID, score: 10, scoreBucket: "promoter" });
    const { result } = renderHook(() => useSurveyWidgetState(RESPONSE_ID, onDone));

    act(() => result.current.selectScore(10));
    await act(async () => result.current.send());
    act(() => jest.advanceTimersByTime(2800));

    expect(onDone).toHaveBeenCalledTimes(1);
    jest.useRealTimers();
  });
});

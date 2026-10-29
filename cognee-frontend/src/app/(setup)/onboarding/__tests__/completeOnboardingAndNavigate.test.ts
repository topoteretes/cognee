const mockShow = jest.fn();
const mockCaptureException = jest.fn();

jest.mock("@mantine/notifications", () => ({
  notifications: { show: (...args: unknown[]) => mockShow(...args) },
}));
jest.mock("@/utils/monitoring", () => ({
  captureException: (...args: unknown[]) => mockCaptureException(...args),
}));

import { completeOnboardingAndNavigate } from "../completeOnboardingAndNavigate";

describe("completeOnboardingAndNavigate", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("persists welcome before completing onboarding and navigating", async () => {
    let resolveWelcome: (() => void) | undefined;
    const markWelcomeSeen = jest.fn(() => new Promise<void>((resolve) => { resolveWelcome = resolve; }));
    const markOnboardingComplete = jest.fn().mockResolvedValue(undefined);
    const navigate = jest.fn();

    const completion = completeOnboardingAndNavigate(markWelcomeSeen, markOnboardingComplete, navigate);

    expect(markWelcomeSeen).toHaveBeenCalledTimes(1);
    expect(markOnboardingComplete).not.toHaveBeenCalled();
    expect(navigate).not.toHaveBeenCalled();

    resolveWelcome?.();
    await completion;

    expect(markOnboardingComplete).toHaveBeenCalledTimes(1);
    expect(navigate).toHaveBeenCalledTimes(1);
  });

  it("does not navigate when the welcome write fails", async () => {
    const error = new Error("welcome write failed");
    const markWelcomeSeen = jest.fn().mockRejectedValue(error);
    const markOnboardingComplete = jest.fn().mockResolvedValue(undefined);
    const navigate = jest.fn();

    await completeOnboardingAndNavigate(markWelcomeSeen, markOnboardingComplete, navigate);

    expect(markOnboardingComplete).not.toHaveBeenCalled();
    expect(navigate).not.toHaveBeenCalled();
    expect(mockCaptureException).toHaveBeenCalledWith(error, { context: "onboarding-completion" });
  });

  // None of the buttons that call this disables itself.
  it("does the writes and the navigation once for a double click", async () => {
    let resolveWelcome: (() => void) | undefined;
    const markWelcomeSeen = jest.fn(() => new Promise<void>((resolve) => { resolveWelcome = resolve; }));
    const markOnboardingComplete = jest.fn().mockResolvedValue(undefined);
    const navigate = jest.fn();

    const first = completeOnboardingAndNavigate(markWelcomeSeen, markOnboardingComplete, navigate);
    const second = completeOnboardingAndNavigate(markWelcomeSeen, markOnboardingComplete, navigate);
    resolveWelcome?.();
    await Promise.all([first, second]);

    expect(markWelcomeSeen).toHaveBeenCalledTimes(1);
    expect(markOnboardingComplete).toHaveBeenCalledTimes(1);
    expect(navigate).toHaveBeenCalledTimes(1);
  });

  it("can be tried again after a failure", async () => {
    const markWelcomeSeen = jest.fn().mockRejectedValueOnce(new Error("offline")).mockResolvedValue(undefined);
    const markOnboardingComplete = jest.fn().mockResolvedValue(undefined);
    const navigate = jest.fn();

    await completeOnboardingAndNavigate(markWelcomeSeen, markOnboardingComplete, navigate);
    await completeOnboardingAndNavigate(markWelcomeSeen, markOnboardingComplete, navigate);

    expect(navigate).toHaveBeenCalledTimes(1);
  });
});

const mockUsePathname = jest.fn();
jest.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
}));

const mockUseUser = jest.fn();
jest.mock("@/modules/users/UserContext", () => ({
  useUser: () => mockUseUser(),
}));

const mockCheckSurveyEligibility = jest.fn();
jest.mock("@/modules/survey/checkSurveyEligibility", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockCheckSurveyEligibility(...args),
}));

const mockAcquireLock = jest.fn();
jest.mock("@/modules/survey/surveyCrossTabLock", () => ({
  acquireCrossTabSurveyLock: (...args: unknown[]) => mockAcquireLock(...args),
}));

jest.mock("@/ui/elements/SurveyWidget", () => ({
  __esModule: true,
  default: ({ responseId }: { responseId: string }) => <div data-testid="survey-widget">{responseId}</div>,
}));

import { render, screen, waitFor } from "@testing-library/react";
import { notifySurveyTrigger } from "@/services/survey/surveyTriggerBridge";
import SurveyProvider from "../SurveyProvider";

const NOW = new Date("2026-07-28T00:00:00.000Z").getTime();

function userMeWithAccountAge(daysOld: number) {
  return { accountCreatedAt: new Date(NOW - daysOld * 24 * 60 * 60 * 1000).toISOString() };
}

describe("SurveyProvider", () => {
  let nowSpy: jest.SpyInstance;

  beforeEach(() => {
    nowSpy = jest.spyOn(Date, "now").mockImplementation(() => NOW);
    mockUsePathname.mockReturnValue("/dashboard");
    mockAcquireLock.mockReturnValue(true);
    mockCheckSurveyEligibility.mockResolvedValue({ eligible: false, responseId: null });
  });

  afterEach(() => {
    nowSpy.mockRestore();
    mockUseUser.mockReset();
    mockCheckSurveyEligibility.mockReset();
    mockAcquireLock.mockReset();
  });

  it("does not check eligibility for a fresh account with no trigger event", () => {
    mockUseUser.mockReturnValue({ userMe: userMeWithAccountAge(1) });

    render(<SurveyProvider>child</SurveyProvider>);

    expect(mockCheckSurveyEligibility).not.toHaveBeenCalled();
  });

  it("checks eligibility with the account-age trigger once the account is 15+ days old", async () => {
    mockUseUser.mockReturnValue({ userMe: userMeWithAccountAge(15) });

    render(<SurveyProvider>child</SurveyProvider>);

    await waitFor(() =>
      expect(mockCheckSurveyEligibility).toHaveBeenCalledWith(
        expect.objectContaining({ trigger: "account_age_15_days", page: "/dashboard" }),
      ),
    );
  });

  it("checks eligibility with the datasource-added trigger when the bridge fires", async () => {
    mockUseUser.mockReturnValue({ userMe: userMeWithAccountAge(1) });

    render(<SurveyProvider>child</SurveyProvider>);
    notifySurveyTrigger("datasource_added");

    await waitFor(() =>
      expect(mockCheckSurveyEligibility).toHaveBeenCalledWith(expect.objectContaining({ trigger: "datasource_added" })),
    );
  });

  it("never fires the eligibility check twice, even when both triggers occur", async () => {
    mockUseUser.mockReturnValue({ userMe: userMeWithAccountAge(20) });

    render(<SurveyProvider>child</SurveyProvider>);
    await waitFor(() => expect(mockCheckSurveyEligibility).toHaveBeenCalledTimes(1));

    notifySurveyTrigger("datasource_added");

    expect(mockCheckSurveyEligibility).toHaveBeenCalledTimes(1);
  });

  it("does not fire the eligibility check when the cross-tab lock is held elsewhere", () => {
    mockAcquireLock.mockReturnValue(false);
    mockUseUser.mockReturnValue({ userMe: userMeWithAccountAge(20) });

    render(<SurveyProvider>child</SurveyProvider>);

    expect(mockCheckSurveyEligibility).not.toHaveBeenCalled();
  });

  it("renders the survey widget once the backend reports the user eligible", async () => {
    mockCheckSurveyEligibility.mockResolvedValue({ eligible: true, responseId: "resp-1" });
    mockUseUser.mockReturnValue({ userMe: userMeWithAccountAge(20) });

    render(<SurveyProvider>child</SurveyProvider>);

    expect(await screen.findByTestId("survey-widget")).toHaveTextContent("resp-1");
  });

  it("does not render the widget when the account is old enough but the backend reports ineligible", async () => {
    mockUseUser.mockReturnValue({ userMe: userMeWithAccountAge(20) });

    render(<SurveyProvider>child</SurveyProvider>);
    await waitFor(() => expect(mockCheckSurveyEligibility).toHaveBeenCalledTimes(1));

    expect(screen.queryByTestId("survey-widget")).not.toBeInTheDocument();
  });
});

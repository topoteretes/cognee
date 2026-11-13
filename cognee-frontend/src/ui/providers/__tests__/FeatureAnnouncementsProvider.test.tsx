import { render, screen } from "@testing-library/react";

const mockUseUser = jest.fn();
jest.mock("@/modules/users/UserContext", () => ({
  useUser: () => mockUseUser(),
}));

jest.mock("@/modules/featureAnnouncements/featureAnnouncementContent", () => ({
  FEATURE_ANNOUNCEMENT_CONTENT: { "new-feature": { title: "New feature" } },
}));

jest.mock("@/ui/elements/FeatureAnnouncementModal", () => ({
  __esModule: true,
  default: () => <div>announcement</div>,
}));

import FeatureAnnouncementsProvider from "../FeatureAnnouncementsProvider";

function renderWith(userMe: { isSeenWelcome: boolean; pendingFeatureAnnouncements: string[] } | null) {
  mockUseUser.mockReturnValue({ userMe, dismissFeatureAnnouncement: jest.fn() });
  render(<FeatureAnnouncementsProvider>child</FeatureAnnouncementsProvider>);
}

afterEach(() => {
  jest.clearAllMocks();
});

describe("FeatureAnnouncementsProvider", () => {
  it("shows a pending announcement to a user who has been through welcome", () => {
    renderWith({ isSeenWelcome: true, pendingFeatureAnnouncements: ["new-feature"] });

    expect(screen.getByText("announcement")).toBeInTheDocument();
  });

  it("holds it back from a brand-new account that has not been through welcome", () => {
    // The (app) layout redirects such a user server-side, but when that check
    // has no answer the app renders without redirecting.
    renderWith({ isSeenWelcome: false, pendingFeatureAnnouncements: ["new-feature"] });

    expect(screen.queryByText("announcement")).not.toBeInTheDocument();
  });

  it("shows nothing while the profile is unknown", () => {
    renderWith(null);

    expect(screen.queryByText("announcement")).not.toBeInTheDocument();
  });
});

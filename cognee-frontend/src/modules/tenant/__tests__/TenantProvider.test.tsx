import { render, screen } from "@testing-library/react";

const mockUsePathname = jest.fn();
jest.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
}));

const mockUseUser = jest.fn();
jest.mock("@/modules/users/UserContext", () => ({
  useUser: () => mockUseUser(),
}));

const mockUseTenantInit = jest.fn();
jest.mock("../useTenantInit", () => ({
  __esModule: true,
  default: () => mockUseTenantInit(),
}));

const mockUseWorkspaceCreation = jest.fn();
jest.mock("../useWorkspaceCreation", () => ({
  __esModule: true,
  default: () => mockUseWorkspaceCreation(),
}));

import { TenantProvider } from "../TenantProvider";

const BASE_INIT_STATE = {
  tenant: null,
  cogniInstance: null,
  serviceUrl: null,
  apiKey: "",
  isInitializing: false,
  tenantReady: true,
  podUnreachable: false,
  error: null,
  releaseLoader: () => {},
};

const BASE_WORKSPACE_STATE = {
  requestCreateWorkspace: () => {},
  nameModalOpen: false,
  nameInput: "",
  setNameInput: () => {},
  submittingName: false,
  nameModalError: null,
  submitWorkspaceName: () => {},
  closeNameModal: () => {},
  isCreatingWorkspace: false,
  creatingWorkspaceName: "",
};

beforeEach(() => {
  jest.clearAllMocks();
  mockUsePathname.mockReturnValue("/dashboard");
  mockUseTenantInit.mockReturnValue(BASE_INIT_STATE);
  mockUseWorkspaceCreation.mockReturnValue(BASE_WORKSPACE_STATE);
});

describe("TenantProvider — /me error handling", () => {
  it("shows a retryable error card instead of an infinite spinner when /me fails", () => {
    // Regression: /api/me uses retry:false and defaults its data to null on
    // error, so a transient failure previously left userMe permanently null
    // with no way to distinguish it from "still loading" — every consumer
    // gated on userMe === null hung behind a spinner with no error and no
    // retry.
    mockUseUser.mockReturnValue({ userMe: null, isUserMeError: true });

    render(<TenantProvider>content</TenantProvider>);

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText("Could not load your account. Please try again.")).toBeInTheDocument();
    expect(screen.getByText("Try again")).toBeInTheDocument();
    expect(screen.getByText("Sign out")).toBeInTheDocument();
  });

  it("shows the plain loading spinner (not an error) while /me is still genuinely loading", () => {
    mockUseUser.mockReturnValue({ userMe: null, isUserMeError: false });

    render(<TenantProvider>content</TenantProvider>);

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
    expect(screen.queryByText("content")).not.toBeInTheDocument();
  });

  it("does not gate on a /me error on a sessionless path", () => {
    mockUsePathname.mockReturnValue("/waitlist");
    mockUseUser.mockReturnValue({ userMe: null, isUserMeError: true });

    render(<TenantProvider>content</TenantProvider>);

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
    expect(screen.getByText("content")).toBeInTheDocument();
  });

  it("renders children once /me resolves successfully", () => {
    mockUseUser.mockReturnValue({
      userMe: { isSeenWelcome: true, onboardingCompletedAt: "2026-01-01", activeWorkspaceId: "t1", userId: "auth0|user-A" },
      isUserMeError: false,
    });

    render(<TenantProvider>content</TenantProvider>);

    expect(screen.getByText("content")).toBeInTheDocument();
  });
});

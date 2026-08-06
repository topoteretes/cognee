import { renderHook } from "@testing-library/react";

const mockReplace = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
}));

const mockUseCogniInstance = jest.fn();
const mockUseTenant = jest.fn();
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => mockUseCogniInstance(),
  useTenant: () => mockUseTenant(),
}));

const mockUseUser = jest.fn();
jest.mock("@/modules/users/UserContext", () => ({
  useUser: () => mockUseUser(),
}));

import { useOnboardingRedirect } from "../useOnboardingRedirect";

const TENANT = { tenant_id: "t1", tenant_name: "Personal" };

beforeEach(() => {
  jest.clearAllMocks();
  mockUseCogniInstance.mockReturnValue({ isInitializing: false });
  mockUseTenant.mockReturnValue({ tenant: TENANT });
});

describe("useOnboardingRedirect", () => {
  it("redirects to /onboarding when the user has never completed it", () => {
    mockUseUser.mockReturnValue({ userMe: { onboardingCompletedAt: null } });

    renderHook(() => useOnboardingRedirect());

    expect(mockReplace).toHaveBeenCalledWith("/onboarding");
  });

  it("does not redirect once onboardingCompletedAt is set", () => {
    mockUseUser.mockReturnValue({ userMe: { onboardingCompletedAt: "2026-07-01T00:00:00Z" } });

    renderHook(() => useOnboardingRedirect());

    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("does not redirect while /me hasn't resolved yet (userMe === null)", () => {
    mockUseUser.mockReturnValue({ userMe: null });

    renderHook(() => useOnboardingRedirect());

    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("does not redirect while tenant provisioning is still in flight", () => {
    mockUseTenant.mockReturnValue({ tenant: null });
    mockUseUser.mockReturnValue({ userMe: { onboardingCompletedAt: null } });

    renderHook(() => useOnboardingRedirect());

    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("does not redirect while cogniInstance is still initializing", () => {
    mockUseCogniInstance.mockReturnValue({ isInitializing: true });
    mockUseUser.mockReturnValue({ userMe: { onboardingCompletedAt: null } });

    renderHook(() => useOnboardingRedirect());

    expect(mockReplace).not.toHaveBeenCalled();
  });
});

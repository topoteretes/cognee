import { render, screen } from "@testing-library/react";

const mockUsePathname = jest.fn();
jest.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
}));

const mockUseTenant = jest.fn();
jest.mock("../TenantContext", () => ({
  useTenant: () => mockUseTenant(),
}));

import AfterTenantInit from "../AfterTenantInit";

afterEach(() => {
  jest.clearAllMocks();
});

function renderAt(pathname: string, isInitializing: boolean) {
  mockUsePathname.mockReturnValue(pathname);
  mockUseTenant.mockReturnValue({ isInitializing });
  render(<AfterTenantInit>content</AfterTenantInit>);
}

describe("AfterTenantInit", () => {
  it("holds children back on an app route while tenant init is running", () => {
    renderAt("/datasets", true);

    expect(screen.queryByText("content")).not.toBeInTheDocument();
  });

  it("renders children on an app route once tenant init has settled", () => {
    renderAt("/datasets", false);

    expect(screen.getByText("content")).toBeInTheDocument();
  });

  it.each(["/dashboard", "/welcome", "/onboarding", "/onboarding/window", "/sign-in", "/waitlist"])(
    "renders children on %s while tenant init is running, as those routes never waited",
    (pathname) => {
      renderAt(pathname, true);

      expect(screen.getByText("content")).toBeInTheDocument();
    },
  );
});

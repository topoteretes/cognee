import { render } from "@testing-library/react";

// Only the name the header shows is under test, so everything around it is a stub.
const mockProfileMenu = jest.fn((props: { userName: string; userEmail: string }) => {
  void props;
  return null;
});
jest.mock("../ProfileMenu", () => ({ __esModule: true, default: (props: { userName: string; userEmail: string }) => mockProfileMenu(props) }));
jest.mock("../HelpMenu", () => ({ __esModule: true, default: () => null }));
jest.mock("../WorkspaceBadge", () => ({ __esModule: true, default: () => null }));
jest.mock("next/navigation", () => ({ usePathname: () => "/dashboard" }));
jest.mock("next/image", () => ({ __esModule: true, default: () => null }));
jest.mock("../FilterContext", () => ({
  useFilter: () => ({
    workspace: { id: "ws-1", name: "Personal Workspace", initial: "P", color: "#6510F4", type: "personal" },
    workspaces: [],
    setWorkspace: jest.fn(),
  }),
}));
jest.mock("@/modules/tenant/TenantContext", () => ({
  useTenant: () => ({
    requestCreateWorkspace: jest.fn(),
    availableTenants: [],
    wakeState: "idle",
    tenant: null,
  }),
}));
jest.mock("@/modules/tenant/isTenantSuspended", () => ({ __esModule: true, default: () => false }));
jest.mock("@/modules/users/getLocalUser", () => ({ __esModule: true, default: async () => null }));

let mockCloud = true;
jest.mock("@/utils/isCloudEnvironment", () => ({ __esModule: true, default: () => mockCloud }));

let mockSignIn: { name: string; email: string } | undefined;
jest.mock("@/modules/users/useCurrentUser", () => ({ useCurrentUser: () => ({ data: mockSignIn }) }));

let mockProfile: { firstName: string | null; lastName: string | null } | null | undefined;
jest.mock("@/modules/users/UserContext", () => ({
  useUser: () => ({ userMe: mockProfile === undefined ? null : { profile: mockProfile } }),
}));

import TopBar from "../TopBar";

const shownName = () => mockProfileMenu.mock.calls.at(-1)?.[0].userName;

describe("TopBar profile menu name", () => {
  beforeEach(() => {
    mockCloud = true;
    mockSignIn = { name: "From Auth0", email: "ada@example.com" };
    mockProfile = undefined;
    mockProfileMenu.mockClear();
  });

  it("shows the name the person set on their profile", () => {
    mockProfile = { firstName: "Ada", lastName: "Lovelace" };

    render(<TopBar />);

    expect(shownName()).toBe("Ada Lovelace");
  });

  it("shows one name when only one is set", () => {
    mockProfile = { firstName: null, lastName: "Lovelace" };

    render(<TopBar />);

    expect(shownName()).toBe("Lovelace");
  });

  it("falls back to the sign-in provider's name for an account with no name set", () => {
    mockProfile = { firstName: null, lastName: null };

    render(<TopBar />);

    expect(shownName()).toBe("From Auth0");
  });

  it("falls back to it while /me has not answered, and on a backend without the field", () => {
    mockProfile = undefined;
    render(<TopBar />);
    expect(shownName()).toBe("From Auth0");

    mockProfileMenu.mockClear();
    mockProfile = null;
    render(<TopBar />);
    expect(shownName()).toBe("From Auth0");
  });

  it("shows an empty name rather than the email when there is nothing to show", () => {
    mockSignIn = undefined;
    mockProfile = { firstName: null, lastName: null };

    render(<TopBar />);

    expect(shownName()).toBe("");
  });

  it("leaves the local, signed-out-of-the-cloud user alone", () => {
    mockCloud = false;
    mockProfile = { firstName: "Ada", lastName: "Lovelace" };

    render(<TopBar />);

    // The profile is a cloud account fact; in local mode the name is the local
    // user's, which getLocalUser supplies.
    expect(shownName()).not.toBe("Ada Lovelace");
  });
});

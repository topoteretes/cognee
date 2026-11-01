import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

const mockPush = jest.fn();
const mockMarkWelcomeSeen = jest.fn<Promise<void>, []>();
const mockShowNotification = jest.fn();

jest.mock("next/navigation", () => ({ useRouter: () => ({ push: mockPush }) }));
jest.mock("@mantine/notifications", () => ({ notifications: { show: (...args: unknown[]) => mockShowNotification(...args) } }));
jest.mock("@/modules/tenant/TenantProvider", () => ({ useTenant: () => ({ releaseLoader: jest.fn() }) }));
jest.mock("@/modules/users/UserContext", () => ({ useUser: () => ({ markWelcomeSeen: mockMarkWelcomeSeen }) }));
jest.mock("@/modules/analytics", () => ({ trackEvent: jest.fn() }));
jest.mock("@/ui/elements/Auth/TetrisBackground", () => ({ __esModule: true, default: () => null }));

import WelcomePage from "../WelcomePage";

beforeEach(() => jest.clearAllMocks());

describe("WelcomePage", () => {
  it("continues to onboarding once welcome is saved", async () => {
    mockMarkWelcomeSeen.mockResolvedValue(undefined);
    render(<WelcomePage />);

    fireEvent.click(screen.getByRole("button", { name: "Get started →" }));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/onboarding"));
  });

  // CLO-1110: the save is a network round trip, and a push issued after it
  // lands on whatever page the user went to in the meantime.
  it("does not push to onboarding if the user left before the save finished", async () => {
    let finishSave: () => void = () => {};
    mockMarkWelcomeSeen.mockReturnValue(new Promise((done) => { finishSave = done; }));
    const { unmount } = render(<WelcomePage />);

    fireEvent.click(screen.getByRole("button", { name: "Get started →" }));
    unmount();
    await act(async () => finishSave());

    expect(mockPush).not.toHaveBeenCalled();
  });

  // The page has no other way forward, so a failed save that left the button
  // disabled stranded a brand-new account here until it reloaded.
  it("lets the user try again after the save fails", async () => {
    mockMarkWelcomeSeen.mockRejectedValueOnce(new Error("unavailable")).mockResolvedValueOnce(undefined);
    render(<WelcomePage />);

    fireEvent.click(screen.getByRole("button", { name: "Get started →" }));
    await waitFor(() => expect(mockShowNotification).toHaveBeenCalledWith(expect.objectContaining({ color: "red" })));
    fireEvent.click(screen.getByRole("button", { name: "Get started →" }));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/onboarding"));
    expect(mockMarkWelcomeSeen).toHaveBeenCalledTimes(2);
  });
});

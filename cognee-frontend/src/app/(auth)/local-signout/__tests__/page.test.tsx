/** @jest-environment-options {"url": "https://ui.example.test/local-signout"} */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MantineProvider } from "@mantine/core";
import LocalSignOutPage from "../page";
import { navigateTo } from "@/utils/navigateTo";

jest.mock("@/utils/navigateTo", () => ({ navigateTo: jest.fn() }));
// The canvas animation behind the auth pages; jsdom has no canvas.
jest.mock("@/ui/elements/Auth/TetrisBackground", () => () => null);

// The refresh cookie is sent to the backend's auth routes only, so signing out
// has to be a request from the browser to the backend. Through the UI server,
// as it used to go, the backend would never see the cookie and the login
// would survive the sign-out.
//
// For the same reason only the backend can end the login: its cookies are not
// readable by the page. A sign-out it did not confirm leaves the user signed in.

const renderPage = () => render(<MantineProvider><LocalSignOutPage /></MantineProvider>);

describe("local sign-out", () => {
  beforeEach(() => fetchMock.resetMocks());

  it("asks the backend to end the login, with the browser's cookies, then opens the login page", async () => {
    fetchMock.mockResponseOnce("{}");

    renderPage();

    await waitFor(() => expect(navigateTo).toHaveBeenCalledWith("/local-login"));
    expect(fetchMock.mock.calls).toEqual([
      ["https://ui.example.test:8000/api/v1/auth/logout", { method: "POST", credentials: "include" }],
    ]);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("opens the login page only after the backend has answered", async () => {
    let answer!: () => void;
    fetchMock.mockResponseOnce(() => new Promise((resolve) => { answer = () => resolve("{}"); }));

    renderPage();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(navigateTo).not.toHaveBeenCalled();
    answer();
    await waitFor(() => expect(navigateTo).toHaveBeenCalledWith("/local-login"));
  });

  it.each([
    ["answers with an error", () => fetchMock.mockResponseOnce("Internal Server Error", { status: 500 })],
    ["cannot be reached", () => fetchMock.mockRejectOnce(new TypeError("Failed to fetch"))],
  ])("stays, and says the user is still signed in, when the backend %s", async (_case, backend) => {
    backend();

    renderPage();

    expect((await screen.findByRole("alert")).textContent).toContain("You are still signed in");
    expect(navigateTo).not.toHaveBeenCalled();
  });

  it("signs out when the user tries again and the backend confirms it", async () => {
    fetchMock.mockResponses(["", { status: 500 }], ["{}", { status: 200 }]);

    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));

    await waitFor(() => expect(navigateTo).toHaveBeenCalledWith("/local-login"));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

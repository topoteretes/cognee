/** @jest-environment-options {"url": "https://ui.example.test/local-signout"} */
import { render, waitFor } from "@testing-library/react";
import LocalSignOutPage from "../page";
import { navigateTo } from "@/utils/navigateTo";

jest.mock("@/utils/navigateTo", () => ({ navigateTo: jest.fn() }));

// The refresh cookie is sent to the backend's auth routes only, so signing out
// has to be a request from the browser to the backend. Through the UI server,
// as it used to go, the backend would never see the cookie and the login
// would survive the sign-out.

describe("local sign-out", () => {
  beforeEach(() => fetchMock.resetMocks());

  it("asks the backend to end the login, with the browser's cookies, then opens the login page", async () => {
    fetchMock.mockResponseOnce("{}");

    render(<LocalSignOutPage />);

    await waitFor(() => expect(navigateTo).toHaveBeenCalledWith("/local-login"));
    expect(fetchMock.mock.calls).toEqual([
      ["https://ui.example.test:8000/api/v1/auth/logout", { method: "POST", credentials: "include" }],
    ]);
  });

  it("opens the login page only after the backend has answered", async () => {
    let answer!: () => void;
    fetchMock.mockResponseOnce(() => new Promise((resolve) => { answer = () => resolve("{}"); }));

    render(<LocalSignOutPage />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(navigateTo).not.toHaveBeenCalled();
    answer();
    await waitFor(() => expect(navigateTo).toHaveBeenCalledWith("/local-login"));
  });

  it("still opens the login page when the backend is down", async () => {
    fetchMock.mockRejectOnce(new TypeError("Failed to fetch"));

    render(<LocalSignOutPage />);

    await waitFor(() => expect(navigateTo).toHaveBeenCalledWith("/local-login"));
  });
});

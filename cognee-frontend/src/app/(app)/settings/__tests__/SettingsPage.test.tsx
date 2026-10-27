import { render, screen } from "@testing-library/react";
import SettingsPage from "../SettingsPage";

let mockCloud = false;
jest.mock("@/utils/isCloudEnvironment", () => ({ __esModule: true, default: () => mockCloud }));
jest.mock("@/modules/analytics", () => ({ TrackPageView: () => null, trackPageEvent: jest.fn() }));
jest.mock("../elements/ProfileWidget", () => ({ __esModule: true, default: () => <div>Editable profile</div> }));

it("shows local account information and uses local sign-out", () => {
  mockCloud = false;
  render(<SettingsPage />);
  expect(screen.getByRole("heading", { name: "Account" })).toBeInTheDocument();
  expect(screen.queryByText("Editable profile")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Log out" })).toHaveAttribute("href", "/api/local-signout");
});

it("preserves the Cloud profile and sign-out route", () => {
  mockCloud = true;
  render(<SettingsPage />);
  expect(screen.getByText("Editable profile")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Log out" })).toHaveAttribute("href", "/api/signout");
});

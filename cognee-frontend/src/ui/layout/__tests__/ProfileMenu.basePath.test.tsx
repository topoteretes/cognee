import { fireEvent, render, screen } from "@testing-library/react";
import ProfileMenu from "../ProfileMenu";
const original = process.env.NEXT_PUBLIC_UI_BASE_PATH;
afterEach(() => {
  if (original === undefined) delete process.env.NEXT_PUBLIC_UI_BASE_PATH;
  else process.env.NEXT_PUBLIC_UI_BASE_PATH = original;
});
it.each(["", "/cognee"])("mounts the raw logout link under %s without manually prefixing Next Link", (base) => {
  process.env.NEXT_PUBLIC_UI_BASE_PATH = base;
  render(<ProfileMenu userName="Test" userEmail="test@example.com" logoutHref="/api/local-signout" />);
  fireEvent.click(screen.getByRole("button", { name: "Account menu" }));
  expect(screen.getByRole("link", { name: "Log out" })).toHaveAttribute("href", `${base}/api/local-signout`);
});

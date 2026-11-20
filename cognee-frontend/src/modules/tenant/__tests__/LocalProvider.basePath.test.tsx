import { render, screen } from "@testing-library/react";
import { LocalProvider } from "../LocalProvider";

const original = process.env.NEXT_PUBLIC_UI_BASE_PATH;
afterEach(() => {
  if (original === undefined) delete process.env.NEXT_PUBLIC_UI_BASE_PATH;
  else process.env.NEXT_PUBLIC_UI_BASE_PATH = original;
});
it.each(["", "/cognee", "/tools/cognee"])("does not probe auth or redirect again on %s/local-login", (base) => {
  process.env.NEXT_PUBLIC_UI_BASE_PATH = base;
  window.history.replaceState({}, "", `${base}/local-login`);
  fetchMock.resetMocks();
  render(<LocalProvider><span>Login form</span></LocalProvider>);
  expect(screen.getByText("Login form")).toBeInTheDocument();
  expect(fetchMock).not.toHaveBeenCalled();
});

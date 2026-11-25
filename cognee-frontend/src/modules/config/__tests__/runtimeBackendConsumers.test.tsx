/** @jest-environment-options {"url": "https://ui.example.test/dashboard"} */
import { useContext, type ComponentProps, type ReactNode } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { LocalProvider } from "@/modules/tenant/LocalProvider";
import { TenantContext } from "@/modules/tenant/TenantContext";
import LocalSignInForm from "@/app/(auth)/local-login/partials/LocalSignInForm";
import nextConfig from "../../../../next.config";
import { getLocalApiUrl } from "@/modules/users/getLocalApiUrl";

jest.mock("@mantine/core", () => ({
  Flex: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  Text: ({ children }: { children: ReactNode }) => <p>{children}</p>,
  Title: ({ children }: { children: ReactNode }) => <h1>{children}</h1>,
  TextInput: ({ label, value, onChange }: ComponentProps<"input"> & { label: string }) =>
    <label>{label}<input value={value} onChange={onChange} /></label>,
  PasswordInput: ({ label, value, onChange }: ComponentProps<"input"> & { label: string }) =>
    <label>{label}<input type="password" value={value} onChange={onChange} /></label>,
  Button: ({ children, type }: ComponentProps<"button">) => <button type={type}>{children}</button>,
}));

const originalUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL;
beforeEach(() => {
  fetchMock.resetMocks();
  const config = document.createElement("script");
  config.id = "cognee-runtime-config";
  config.type = "application/json";
  config.textContent = JSON.stringify({ backendUrl: "https://runtime.example.test" });
  document.head.appendChild(config);
});
afterEach(() => {
  document.getElementById("cognee-runtime-config")?.remove();
  if (originalUrl === undefined) delete process.env.NEXT_PUBLIC_LOCAL_API_URL;
  else process.env.NEXT_PUBLIC_LOCAL_API_URL = originalUrl;
});

function ServiceUrl() {
  return <span>{useContext(TenantContext).serviceUrl}</span>;
}

it("checks auth and publishes the same runtime service URL", async () => {
  fetchMock.mockResponseOnce("{}");
  render(<LocalProvider><ServiceUrl /></LocalProvider>);
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
    "https://runtime.example.test/api/v1/users/me", { credentials: "include" },
  ));
  expect(screen.getByText("https://runtime.example.test")).toBeInTheDocument();
});

it("sends the entered login credentials to the runtime backend", async () => {
  // A rejected login avoids jsdom navigation while exercising the request.
  fetchMock.mockResponseOnce(JSON.stringify({ detail: "LOGIN_BAD_CREDENTIALS" }), { status: 401 });
  const { container } = render(<LocalSignInForm />);
  fireEvent.change(screen.getByLabelText("Email"), { target: { value: "typed@example.test" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "typed&password" } });
  const form = container.querySelector("form");
  if (!form) throw new Error("Login form was not rendered");
  fireEvent.submit(form);
  await screen.findByText("Invalid email or password.");
  expect(fetchMock.mock.calls[0][0]).toBe("https://runtime.example.test/api/v1/auth/login");
  const body = new URLSearchParams(String(fetchMock.mock.calls[0][1]?.body));
  expect(body.get("username")).toBe("typed@example.test");
  expect(body.get("password")).toBe("typed&password");
});

it("does not bake a default URL that defeats browser-host resolution", () => {
  document.getElementById("cognee-runtime-config")?.remove();
  delete process.env.NEXT_PUBLIC_LOCAL_API_URL;
  const bakedUrl = nextConfig.env?.NEXT_PUBLIC_LOCAL_API_URL;
  if (bakedUrl !== undefined) process.env.NEXT_PUBLIC_LOCAL_API_URL = bakedUrl;
  expect(getLocalApiUrl()).toBe("https://ui.example.test:8000");
});

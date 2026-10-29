import { renderToString } from "react-dom/server";
import { renderHook } from "@testing-library/react";
import { useCurrentHost } from "@/utils/useCurrentHost";

function Host() {
  return <span>{String(useCurrentHost())}</span>;
}

describe("useCurrentHost", () => {
  it("is the host the browser is on, with the port", () => {
    const { result } = renderHook(() => useCurrentHost());

    expect(result.current).toBe(window.location.host);
  });

  it("is null on the server, where there is no browser to ask", () => {
    expect(renderToString(<Host />)).toContain("null");
  });
});

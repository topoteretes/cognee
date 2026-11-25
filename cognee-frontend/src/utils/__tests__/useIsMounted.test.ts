import { renderHook } from "@testing-library/react";
import { StrictMode } from "react";
import { useIsMounted } from "../useIsMounted";

describe("useIsMounted", () => {
  it("is true while the component is mounted and false once it is gone", () => {
    const { result, unmount } = renderHook(() => useIsMounted());

    expect(result.current()).toBe(true);
    unmount();
    expect(result.current()).toBe(false);
  });

  it("is still true after a StrictMode remount", () => {
    const { result } = renderHook(() => useIsMounted(), { wrapper: StrictMode });

    expect(result.current()).toBe(true);
  });

  it("returns the same function on every render, so it is safe in dependency lists", () => {
    const { result, rerender } = renderHook(() => useIsMounted());
    const first = result.current;

    rerender();

    expect(result.current).toBe(first);
  });
});

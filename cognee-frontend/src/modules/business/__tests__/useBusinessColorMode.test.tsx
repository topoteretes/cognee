import { act, renderHook } from "@testing-library/react";
import { useBusinessColorMode } from "../useBusinessColorMode";

const KEY = "cognee_business_color_mode";

afterEach(() => {
  jest.restoreAllMocks();
  window.localStorage.clear();
});

describe("useBusinessColorMode", () => {
  it("defaults to type mode with nothing stored", () => {
    const { result } = renderHook(() => useBusinessColorMode());
    expect(result.current[0]).toBe("type");
  });

  it("reads a stored source mode on the first render", () => {
    window.localStorage.setItem(KEY, "source");
    const { result } = renderHook(() => useBusinessColorMode());
    expect(result.current[0]).toBe("source");
  });

  it("falls back to type mode when the stored value is not a mode", () => {
    window.localStorage.setItem(KEY, "rainbow");
    const { result } = renderHook(() => useBusinessColorMode());
    expect(result.current[0]).toBe("type");
  });

  it("stores the choice and updates every mounted hook in the same tab", () => {
    const first = renderHook(() => useBusinessColorMode());
    const second = renderHook(() => useBusinessColorMode());
    act(() => first.result.current[1]("source"));
    expect(window.localStorage.getItem(KEY)).toBe("source");
    expect(first.result.current[0]).toBe("source");
    expect(second.result.current[0]).toBe("source");
  });

  it("follows a change made in another tab", () => {
    const { result } = renderHook(() => useBusinessColorMode());
    act(() => {
      window.localStorage.setItem(KEY, "source");
      window.dispatchEvent(new StorageEvent("storage", { key: KEY }));
    });
    expect(result.current[0]).toBe("source");
  });

  it("ignores a storage event for an unrelated key", () => {
    const { result } = renderHook(() => useBusinessColorMode());
    act(() => {
      window.localStorage.setItem(KEY, "source");
      window.dispatchEvent(new StorageEvent("storage", { key: "something_else" }));
    });
    expect(result.current[0]).toBe("type");
  });

  it("still switches for this session when storage refuses the write", () => {
    jest.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota");
    });
    const { result } = renderHook(() => useBusinessColorMode());
    act(() => result.current[1]("source"));
    expect(result.current[0]).toBe("source");
  });
});

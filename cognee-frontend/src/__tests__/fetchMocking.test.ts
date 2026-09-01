/**
 * Guards the jest fetch setup itself.
 *
 * jest.setup.ts previously called `enableMocks()` followed by `dontMock()`,
 * which leaves jest-fetch-mock's isMocking predicate permanently false. The
 * stubbing API still appeared to work, but every `mockResponseOnce` was a no-op
 * and the call fell through to a real network request. These two tests fail if
 * that combination ever comes back.
 */
import fetchMock from "jest-fetch-mock";

describe("jest fetch setup", () => {
  it("provides the fetch API classes jsdom lacks", () => {
    expect(typeof Response).toBe("function");
    expect(typeof Headers).toBe("function");
    expect(typeof Request).toBe("function");
  });

  it("lets a test stub fetch instead of reaching the network", async () => {
    fetchMock.resetMocks();
    fetchMock.mockResponseOnce(JSON.stringify({ stubbed: true }));

    const response = await fetch("http://backend.invalid/api/v1/datasets");

    expect(await response.json()).toEqual({ stubbed: true });
    expect(fetchMock).toHaveBeenCalledWith("http://backend.invalid/api/v1/datasets");
  });
});

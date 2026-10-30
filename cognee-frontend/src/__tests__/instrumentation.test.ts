import { register } from "../instrumentation";

describe("instrumentation register", () => {
  const original = process.env.COGNEE_BACKEND_URL;

  beforeEach(() => {
    jest.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => {
    if (original === undefined) delete process.env.COGNEE_BACKEND_URL;
    else process.env.COGNEE_BACKEND_URL = original;
    jest.restoreAllMocks();
  });

  it("reports an invalid COGNEE_BACKEND_URL without stopping startup", async () => {
    process.env.COGNEE_BACKEND_URL = "cognee:8000";

    await expect(register()).resolves.toBeUndefined();
    expect(console.error).toHaveBeenCalledWith(
      "[cognee-ui] Invalid backend configuration:",
      expect.objectContaining({ message: expect.stringContaining("COGNEE_BACKEND_URL") }),
    );
  });

  it("stays quiet for a valid URL", async () => {
    process.env.COGNEE_BACKEND_URL = "http://backend.test:8000";

    await register();

    expect(console.error).not.toHaveBeenCalled();
  });
});

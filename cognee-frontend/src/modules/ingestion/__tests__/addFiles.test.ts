import addFiles from "@/modules/ingestion/addFiles";
import type { CogneeInstance } from "@/modules/instances/types";

const mockCapture = jest.fn();
jest.mock("@/utils/monitoring", () => ({
  captureException: (...args: unknown[]) => mockCapture(...args),
}));

function makeFile(name = "a.pdf", size = 10): File {
  return new File(["x".repeat(size)], name, { type: "application/pdf" });
}

function jsonResponse(body: unknown): Response {
  return { json: async () => body } as unknown as Response;
}

describe("addFiles", () => {
  beforeEach(() => jest.clearAllMocks());

  it("posts every file plus datasetId/datasetName as multipart to /v1/add", async () => {
    const upload = jest.fn().mockResolvedValue(jsonResponse({ pipeline_run_id: "run-1" }));
    const instance = { name: "t", fetch: jest.fn(), upload } as unknown as CogneeInstance;

    const files = [makeFile("a.pdf"), makeFile("b.pdf")];
    const res = await addFiles({ id: "ds-1", name: "My DS" }, files, instance);

    expect(upload).toHaveBeenCalledTimes(1);
    const [path, makeBody] = upload.mock.calls[0];
    expect(path).toBe("/v1/add");

    const form = makeBody() as FormData;
    expect(form.getAll("data")).toHaveLength(2);
    expect(form.get("datasetId")).toBe("ds-1");
    expect(form.get("datasetName")).toBe("My DS");
    // Upload-only: no cognify knobs, no run_in_background.
    expect(form.get("chunk_size")).toBeNull();
    expect(form.get("run_in_background")).toBeNull();

    expect(res.pipeline_run_id).toBe("run-1");
  });

  it("omits datasetName when only an id is given", async () => {
    const upload = jest.fn().mockResolvedValue(jsonResponse({}));
    const instance = { name: "t", fetch: jest.fn(), upload } as unknown as CogneeInstance;

    await addFiles({ id: "ds-1" }, [makeFile()], instance);
    const form = (upload.mock.calls[0][1] as () => FormData)();
    expect(form.get("datasetId")).toBe("ds-1");
    expect(form.get("datasetName")).toBeNull();
  });

  it("falls back to instance.fetch when upload is unavailable", async () => {
    const fetch = jest.fn().mockResolvedValue(jsonResponse({ pipeline_run_id: "run-2" }));
    const instance = { name: "t", fetch } as unknown as CogneeInstance;

    const res = await addFiles({ id: "ds-1" }, [makeFile()], instance);
    expect(fetch).toHaveBeenCalledWith("/v1/add", expect.objectContaining({ method: "POST" }));
    expect(res.pipeline_run_id).toBe("run-2");
  });

  it("throws on an app-level error in a 200 body", async () => {
    const upload = jest.fn().mockResolvedValue(jsonResponse({ error: "add failed" }));
    const instance = { name: "t", fetch: jest.fn(), upload } as unknown as CogneeInstance;

    await expect(addFiles({ id: "ds-1" }, [makeFile()], instance)).rejects.toThrow("add failed");
    expect(mockCapture).toHaveBeenCalled();
  });

  it("relabels a client timeout as an upload-timeout without blaming file size", async () => {
    const upload = jest.fn().mockRejectedValue(new Error("Request timed out."));
    const instance = { name: "t", fetch: jest.fn(), upload } as unknown as CogneeInstance;

    await expect(addFiles({ id: "ds-1" }, [makeFile()], instance)).rejects.toMatchObject({
      name: "UploadTimeoutError",
    });
  });
});

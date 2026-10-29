const mockShow = jest.fn();
const mockCaptureException = jest.fn();
jest.mock("@mantine/notifications", () => ({ notifications: { show: (...args: unknown[]) => mockShow(...args) } }));
jest.mock("@/utils/monitoring", () => ({ captureException: (...args: unknown[]) => mockCaptureException(...args) }));

import { saveAssignment } from "../saveAssignment";

const base = { label: "Prompt", choice: "Legal", datasetId: "ds-1", datasetName: "Contracts" };

beforeEach(() => jest.clearAllMocks());

describe("saveAssignment", () => {
  it("announces the choice only once it is stored", async () => {
    let finish: () => void = () => {};
    const onSaved = jest.fn();
    const pending = saveAssignment({ ...base, save: () => new Promise((done) => { finish = done; }), onSaved, restore: jest.fn() });

    expect(mockShow).not.toHaveBeenCalled();
    expect(onSaved).not.toHaveBeenCalled();

    finish();
    await pending;

    expect(onSaved).toHaveBeenCalledTimes(1);
    expect(mockShow).toHaveBeenCalledWith(expect.objectContaining({ title: "Prompt updated", color: "green", message: '"Contracts" now uses "Legal".' }));
  });

  it("puts the previous choice back and says the save failed when it is not stored", async () => {
    const restore = jest.fn();
    const onSaved = jest.fn();

    await saveAssignment({ ...base, save: () => Promise.reject(new Error("pod unreachable")), onSaved, restore });

    expect(restore).toHaveBeenCalledTimes(1);
    expect(onSaved).not.toHaveBeenCalled();
    expect(mockShow).toHaveBeenCalledTimes(1);
    expect(mockShow).toHaveBeenCalledWith(expect.objectContaining({ title: "Prompt not confirmed", color: "red" }));
    expect(mockCaptureException).toHaveBeenCalledWith(expect.any(Error), expect.objectContaining({ datasetId: "ds-1", label: "Prompt" }));
  });
});

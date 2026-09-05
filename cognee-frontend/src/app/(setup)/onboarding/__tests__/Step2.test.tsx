export {};

const mockCreateDataset = jest.fn();
const mockRememberData = jest.fn();
const mockPollDatasetStatus = jest.fn();

jest.mock("@/modules/datasets/createDataset", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockCreateDataset(...args),
}));
jest.mock("@/modules/ingestion/rememberData", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockRememberData(...args),
}));
jest.mock("@/modules/datasets/pollDatasetStatus", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockPollDatasetStatus(...args),
}));
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useTenant: () => ({ tenantReady: true }),
  useCogniInstance: () => ({ cogniInstance: { id: "inst" } }),
}));
jest.mock("../useOnboardingTrackEvent", () => ({
  useOnboardingTrackEvent: () => jest.fn(),
}));
jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: jest.fn() }),
}));
jest.mock("@/modules/users/UserContext", () => ({
  useUser: () => ({ markOnboardingComplete: jest.fn() }),
}));
jest.mock("@/utils/browserStorage", () => ({
  setAwaitingDataset: jest.fn(),
  clearAwaitingDataset: jest.fn(),
}));

import { render, waitFor } from "@testing-library/react";
import { Step2 } from "../partials/Step2";

const FILES = [new File(["hello"], "notes.txt", { type: "text/plain" })];
const MORE_FILES = [...FILES, new File(["second"], "second.txt", { type: "text/plain" })];
const keyFor = (files: File[]) => files.map((f) => `${f.name}:${f.size}:${f.lastModified}`).join("|");
const INSTANCE = { id: "inst" } as unknown as Parameters<typeof Step2>[0]["cogniInstance"];

function renderStep2(
  datasetId: string | null,
  onDatasetCreated = jest.fn(),
  { files = FILES, ingestedKey = null as string | null, onIngested = jest.fn() } = {},
) {
  return render(
    <Step2
      files={files}
      datasetId={datasetId}
      onNext={jest.fn()}
      onBack={jest.fn()}
      onDatasetCreated={onDatasetCreated}
      ingestedKey={ingestedKey}
      onIngested={onIngested}
      cogniInstance={INSTANCE}
    />,
  );
}

beforeEach(() => {
  mockCreateDataset.mockResolvedValue({ id: "ds_1" });
  mockRememberData.mockResolvedValue(undefined);
  mockPollDatasetStatus.mockResolvedValue(undefined);
});

describe("Step2 pipeline", () => {
  it("creates the dataset and ingests the files on a first run", async () => {
    const onDatasetCreated = jest.fn();
    renderStep2(null, onDatasetCreated);

    await waitFor(() => expect(mockRememberData).toHaveBeenCalledTimes(1));
    expect(mockCreateDataset).toHaveBeenCalledTimes(1);
    // Reported as soon as it exists, so a remount can reuse it.
    expect(onDatasetCreated).toHaveBeenCalledWith("ds_1");
  });

  // Back from here lands on Step 1, whose Continue remounts this component.
  // Re-running would create a second dataset and pay for a second upload and
  // cognify of the same files.
  it("does not re-create or re-ingest when it remounts with the same file set", async () => {
    renderStep2("ds_1", jest.fn(), { ingestedKey: keyFor(FILES) });

    await waitFor(() => expect(mockPollDatasetStatus).toHaveBeenCalled());
    expect(mockCreateDataset).not.toHaveBeenCalled();
    expect(mockRememberData).not.toHaveBeenCalled();
  });

  // The guard keys on the file set, not on the dataset existing: Step 1 lets
  // files be added and removed after the first run, and keying on the dataset
  // would drop the new selection while the bars filled as if it had been sent.
  it("re-ingests when the user added a file after going Back", async () => {
    renderStep2("ds_1", jest.fn(), { files: MORE_FILES, ingestedKey: keyFor(FILES) });

    await waitFor(() => expect(mockRememberData).toHaveBeenCalledTimes(1));
    expect(mockRememberData.mock.calls[0][1]).toHaveLength(2);
    // Reuses the dataset it already has rather than creating a second one.
    expect(mockCreateDataset).not.toHaveBeenCalled();
  });

  // The signature is recorded after rememberData resolves, so a failed upload
  // stays retryable — Back then Continue is the only retry path there is.
  it("retries the upload when the first attempt failed", async () => {
    mockRememberData.mockRejectedValueOnce(new Error("upload failed"));
    const onIngested = jest.fn();
    const { unmount } = renderStep2(null, jest.fn(), { onIngested });

    await waitFor(() => expect(mockRememberData).toHaveBeenCalledTimes(1));
    expect(onIngested).not.toHaveBeenCalled();
    unmount();

    renderStep2("ds_1", jest.fn(), { ingestedKey: null });
    await waitFor(() => expect(mockRememberData).toHaveBeenCalledTimes(2));
  });
});

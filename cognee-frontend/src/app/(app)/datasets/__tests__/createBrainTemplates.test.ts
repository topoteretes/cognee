import type { CogneeInstance } from "@/modules/instances/types";
import type { GraphModelsConfig } from "@/modules/configuration/userConfiguration";

const mockLoadGraphModelsConfig = jest.fn();
const mockSaveCustomPrompt = jest.fn();
const mockAssignPromptToDataset = jest.fn();
const mockClearDatasetOutdated = jest.fn();
jest.mock("@/modules/configuration/userConfiguration", () => ({
  fetchGraphModelsConfig: (...args: unknown[]) => mockLoadGraphModelsConfig(...args),
  saveCustomPrompt: (...args: unknown[]) => mockSaveCustomPrompt(...args),
  assignPromptToDataset: (...args: unknown[]) => mockAssignPromptToDataset(...args),
  clearDatasetOutdated: (...args: unknown[]) => mockClearDatasetOutdated(...args),
}));

import { applyCreateBrainTemplate, CREATE_BRAIN_TEMPLATES } from "@/app/(app)/datasets/createBrainTemplates";

const DATASET_ID = "dataset-1";
const instance = { name: "test-instance", fetch: jest.fn() } as unknown as CogneeInstance;

function makeConfig(overrides: Partial<GraphModelsConfig> = {}): GraphModelsConfig {
  return { models: [], customPrompts: {}, promptAssignments: {}, ontologyAssignments: {}, outdatedDatasets: [], ...overrides };
}

afterEach(() => jest.clearAllMocks());

describe("applyCreateBrainTemplate", () => {
  it("saves the template's default prompt when no prompt exists under that name yet", async () => {
    mockLoadGraphModelsConfig.mockResolvedValue(makeConfig());

    await applyCreateBrainTemplate(instance, DATASET_ID, "legal");

    const legal = CREATE_BRAIN_TEMPLATES.find((t) => t.key === "legal");
    expect(mockSaveCustomPrompt).toHaveBeenCalledWith(instance, legal?.promptName, legal?.promptText);
  });

  it("assigns the template's prompt name to the dataset", async () => {
    mockLoadGraphModelsConfig.mockResolvedValue(makeConfig());

    await applyCreateBrainTemplate(instance, DATASET_ID, "support-tickets");

    const supportTickets = CREATE_BRAIN_TEMPLATES.find((t) => t.key === "support-tickets");
    expect(mockAssignPromptToDataset).toHaveBeenCalledWith(instance, DATASET_ID, supportTickets?.promptName);
  });

  it("does not overwrite an already-customized prompt saved under the template's name", async () => {
    const documents = CREATE_BRAIN_TEMPLATES.find((t) => t.key === "documents");
    mockLoadGraphModelsConfig.mockResolvedValue(
      makeConfig({ customPrompts: { [documents?.promptName ?? ""]: "A user-edited version of this prompt." } }),
    );

    await applyCreateBrainTemplate(instance, DATASET_ID, "documents");

    expect(mockSaveCustomPrompt).not.toHaveBeenCalled();
    expect(mockAssignPromptToDataset).toHaveBeenCalledWith(instance, DATASET_ID, documents?.promptName);
  });

  // A failed read taken for "no prompt saved under this name" would overwrite
  // a prompt the user had customized.
  it("changes nothing when the stored configuration cannot be read", async () => {
    mockLoadGraphModelsConfig.mockRejectedValue(new Error("pod unreachable"));

    await expect(applyCreateBrainTemplate(instance, DATASET_ID, "legal")).rejects.toThrow("pod unreachable");

    expect(mockSaveCustomPrompt).not.toHaveBeenCalled();
    expect(mockAssignPromptToDataset).not.toHaveBeenCalled();
    expect(mockClearDatasetOutdated).not.toHaveBeenCalled();
  });

  it("does nothing when the template key is unknown", async () => {
    await applyCreateBrainTemplate(instance, DATASET_ID, "not-a-real-template" as never);

    expect(mockLoadGraphModelsConfig).not.toHaveBeenCalled();
    expect(mockSaveCustomPrompt).not.toHaveBeenCalled();
    expect(mockAssignPromptToDataset).not.toHaveBeenCalled();
    expect(mockClearDatasetOutdated).not.toHaveBeenCalled();
  });

  it("clears the outdated flag after assigning, since a brand-new dataset has no build to be outdated from", async () => {
    mockLoadGraphModelsConfig.mockResolvedValue(makeConfig());

    await applyCreateBrainTemplate(instance, DATASET_ID, "legal");

    expect(mockClearDatasetOutdated).toHaveBeenCalledWith(instance, DATASET_ID);
  });
});

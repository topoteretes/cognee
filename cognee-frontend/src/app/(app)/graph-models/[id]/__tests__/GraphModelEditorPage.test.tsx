import { act, fireEvent, render, screen } from "@testing-library/react";
import { emptySchema } from "@/modules/graphModels/schemaReducer";
import type { GraphModel } from "@/modules/graphModels/types";
import type { GraphModelsConfig } from "@/modules/configuration/userConfiguration";

const mockReplace = jest.fn();
const mockPush = jest.fn();
const mockFetchConfig = jest.fn();
type EditedModel = Pick<GraphModel, "id" | "name" | "schema">;
const mockSaveGraphModel = jest.fn<Promise<void>, [unknown, EditedModel, string]>();
const mockDeleteGraphModel = jest.fn<Promise<void>, [unknown, string]>();
const mockInstance = { instanceId: "workspace-1", fetch: jest.fn() };

jest.mock("next/navigation", () => ({ useRouter: () => ({ replace: mockReplace, push: mockPush }) }));
jest.mock("@/modules/tenant/TenantProvider", () => ({ useCogniInstance: () => ({ cogniInstance: mockInstance }) }));
jest.mock("@/ui/layout/FilterContext", () => ({ useFilter: () => ({ datasets: [] }) }));
jest.mock("@/modules/configuration/userConfiguration", () => ({
  fetchGraphModelsConfig: (...args: unknown[]) => mockFetchConfig(...args),
  loadGraphModelsConfig: async () => ({ models: [] }),
  saveGraphModel: (...args: [unknown, EditedModel, string]) => mockSaveGraphModel(...args),
  deleteGraphModel: (...args: [unknown, string]) => mockDeleteGraphModel(...args),
}));
jest.mock("@mantine/notifications", () => ({ notifications: { show: jest.fn() } }));
jest.mock("@/modules/analytics", () => ({ TrackPageView: () => null, trackEvent: jest.fn() }));
jest.mock("@/modules/llm/managementLlmApi", () => ({ inferSchema: jest.fn(), downloadRawData: jest.fn() }));
jest.mock("../components/SchemaGraphPreview", () => ({ __esModule: true, default: () => null }));
jest.mock("../components/AddEntityModal", () => ({ __esModule: true, default: () => null }));

import GraphModelEditorPage from "../GraphModelEditorPage";

const NO_MODELS: GraphModelsConfig = { models: [] };
const OLD_MODEL: GraphModel = {
  id: "old-model",
  name: "Stale model",
  schema: emptySchema(),
  createdAt: "2026-01-01T00:00:00Z",
  updatedAt: "2026-01-01T00:00:00Z",
  status: "draft",
};

beforeEach(() => {
  jest.clearAllMocks();
  sessionStorage.clear();
});

// CLO-1110: the editor used to router.replace("/datasets") whenever the load
// came back empty or failed, including after the user had already left.
describe("GraphModelEditorPage loading", () => {
  it("ignores a response for a model the page has already moved on from", async () => {
    let resolveOld: (config: GraphModelsConfig) => void = () => {};
    mockFetchConfig
      .mockReturnValueOnce(new Promise((done) => { resolveOld = done; }))
      .mockResolvedValueOnce(NO_MODELS);
    const { rerender } = render(<GraphModelEditorPage modelId="old-model" />);

    rerender(<GraphModelEditorPage modelId="new-model" />);
    await screen.findByText("This model could not be found.");
    await act(async () => resolveOld({ models: [OLD_MODEL] }));

    expect(screen.getByText("This model could not be found.")).toBeInTheDocument();
    expect(screen.queryByText(OLD_MODEL.name)).not.toBeInTheDocument();
    expect(mockPush).not.toHaveBeenCalled();
  });

  // A read that fails after the page has moved to another model must not mark
  // the new one as failed.
  it("ignores a failure for a model the page has already moved on from", async () => {
    let failOld: (error: Error) => void = () => {};
    mockFetchConfig
      .mockReturnValueOnce(new Promise((_resolve, reject) => { failOld = reject; }))
      .mockResolvedValueOnce({ models: [{ ...OLD_MODEL, id: "new-model", name: "The new one" }] });
    const { rerender } = render(<GraphModelEditorPage modelId="old-model" />);

    rerender(<GraphModelEditorPage modelId="new-model" />);
    await act(async () => failOld(new Error("late failure")));

    expect(screen.queryByText("Could not load this model. Please try again.")).not.toBeInTheDocument();
  });

  it("goes back to loading while a retry is in flight", async () => {
    let finishRetry: (config: GraphModelsConfig) => void = () => {};
    mockFetchConfig
      .mockRejectedValueOnce(new Error("offline"))
      .mockReturnValueOnce(new Promise((done) => { finishRetry = done; }));
    render(<GraphModelEditorPage modelId="model-1" />);
    await screen.findByText("Could not load this model. Please try again.");

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByRole("status")).toHaveTextContent("Loading model");
    expect(screen.queryByText("Could not load this model. Please try again.")).not.toBeInTheDocument();
    await act(async () => finishRetry(NO_MODELS));
  });

  it("shows a missing model in place instead of redirecting", async () => {
    mockFetchConfig.mockResolvedValue(NO_MODELS);

    render(<GraphModelEditorPage modelId="model-1" />);

    expect(await screen.findByText("This model could not be found.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Back to datasets" })).toHaveAttribute("href", "/datasets");
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("tells a failed read apart from a missing model and lets the user retry", async () => {
    mockFetchConfig.mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(NO_MODELS);
    render(<GraphModelEditorPage modelId="model-1" />);
    await screen.findByText("Could not load this model. Please try again.");

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByText("This model could not be found.")).toBeInTheDocument();
    expect(mockFetchConfig).toHaveBeenCalledTimes(2);
    expect(mockReplace).not.toHaveBeenCalled();
  });

  // CLO-1110: a delete takes a network round trip, and a push issued after it
  // lands on whatever page the user went to in the meantime.
  describe("deleting a model", () => {
    async function clickDelete(): Promise<{ unmount: () => void; finishDelete: () => void }> {
      let finishDelete: () => void = () => {};
      mockFetchConfig.mockResolvedValue({ models: [{ ...OLD_MODEL, id: "model-1" }] });
      mockDeleteGraphModel.mockReturnValue(new Promise((done) => { finishDelete = done; }));
      jest.spyOn(window, "confirm").mockReturnValue(true);
      const { unmount } = render(<GraphModelEditorPage modelId="model-1" />);
      fireEvent.click(await screen.findByTitle("Delete graph model"));
      await act(async () => {});
      return { unmount, finishDelete };
    }

    it("goes back to the datasets once the delete has finished", async () => {
      const { finishDelete } = await clickDelete();

      await act(async () => finishDelete());

      expect(mockPush).toHaveBeenCalledWith("/datasets");
    });

    it("stays where the user went if they left the editor before it finished", async () => {
      const { unmount, finishDelete } = await clickDelete();

      unmount();
      await act(async () => finishDelete());

      expect(mockPush).not.toHaveBeenCalled();
    });
  });

  // The editor hands the edited model to saveGraphModel, which merges it into
  // what is stored at write time and flags its datasets when the schema changed
  // (see userConfiguration.test.ts). These pin what the editor hands over.
  describe("what saving and deleting write", () => {
    async function openEditor(): Promise<void> {
      mockFetchConfig.mockResolvedValueOnce({ models: [{ ...OLD_MODEL, id: "model-1", name: "Edited" }] });
      mockSaveGraphModel.mockResolvedValue(undefined);
      render(<GraphModelEditorPage modelId="model-1" />);
      await screen.findByTitle("Delete graph model");
    }

    it("saving stores this model as it is shown, under its own id", async () => {
      await openEditor();

      await act(async () => { fireEvent.click(screen.getAllByRole("button", { name: /^Save/ })[0]); });

      expect(mockSaveGraphModel).toHaveBeenCalledWith(mockInstance, { id: "model-1", name: "Edited", schema: OLD_MODEL.schema }, expect.any(String));
    });

    it("deleting asks for this model to be deleted, which also flags the datasets that used it", async () => {
      await openEditor();
      mockDeleteGraphModel.mockResolvedValue(undefined);
      jest.spyOn(window, "confirm").mockReturnValue(true);

      await act(async () => { fireEvent.click(screen.getByTitle("Delete graph model")); });

      expect(mockDeleteGraphModel).toHaveBeenCalledWith(expect.anything(), "model-1");
    });

    it("stays on the editor when the delete could not be stored", async () => {
      await openEditor();
      mockDeleteGraphModel.mockRejectedValueOnce(new Error("pod unreachable"));
      jest.spyOn(window, "confirm").mockReturnValue(true);

      await act(async () => { fireEvent.click(screen.getByTitle("Delete graph model")); });

      expect(mockPush).not.toHaveBeenCalled();
    });
  });
});

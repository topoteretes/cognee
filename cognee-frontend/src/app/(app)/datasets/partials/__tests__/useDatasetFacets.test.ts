import { act, renderHook, waitFor } from "@testing-library/react";
import { notifications } from "@mantine/notifications";
import getDatasetAccess from "@/modules/permissions/getDatasetAccess";
import getGroupOverview from "@/modules/permissions/getGroupOverview";
import useDatasetFacets from "@/app/(app)/datasets/partials/useDatasetFacets";
import type { DatasetAccess, GroupOverview } from "@/modules/permissions/types";
import isCloudEnvironment from "@/utils/isCloudEnvironment";

jest.mock("@/utils/isCloudEnvironment");
jest.mock("@/modules/permissions/getDatasetAccess");
jest.mock("@/modules/permissions/getGroupOverview");
jest.mock("@mantine/notifications", () => ({ notifications: { show: jest.fn() } }));
// Stable references across renders: a new object per call would change
// `refresh`'s identity every render and re-trigger its effect in a loop.
// jest-hoist only allows referencing out-of-scope variables here when their
// name starts with "mock" — that's not a style choice, a plain name throws.
const mockCogniInstance = { fetch: jest.fn() };
const mockTenant = { tenant_id: "t-1" };
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: mockCogniInstance }),
  useTenant: () => ({ tenant: mockTenant }),
}));
jest.mock("@/modules/users/useCurrentUser", () => ({
  useCurrentUser: () => ({ data: { email: "me@acme.io" } }),
}));

const ACCESS: DatasetAccess[] = [
  { datasetId: "mine", datasetName: "mine", ownerId: "me", isOwner: true, permissions: ["read", "share"], sources: [], sharedWith: null },
];
const GROUPS: GroupOverview[] = [];

beforeEach(() => {
  jest.mocked(isCloudEnvironment).mockReturnValue(true);
});

afterEach(() => {
  jest.resetAllMocks();
});

describe("useDatasetFacets", () => {
  it("does not request Cloud access or groups in local mode, including refreshes", async () => {
    jest.mocked(isCloudEnvironment).mockReturnValue(false);
    const { result, rerender } = renderHook(({ key }) => useDatasetFacets(key), {
      initialProps: { key: "mine" },
    });

    expect(result.current.status).toBe("ready");
    expect(result.current.accessById.size).toBe(0);
    rerender({ key: "mine,new" });
    await act(async () => { await result.current.refresh(); });

    expect(getDatasetAccess).not.toHaveBeenCalled();
    expect(getGroupOverview).not.toHaveBeenCalled();
    expect(notifications.show).not.toHaveBeenCalled();
  });

  it("starts loading and settles on ready once both reads land", async () => {
    jest.mocked(getDatasetAccess).mockResolvedValue(ACCESS);
    jest.mocked(getGroupOverview).mockResolvedValue(GROUPS);

    const { result } = renderHook(() => useDatasetFacets("mine"));
    expect(result.current.status).toBe("loading");

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.accessById.get("mine")?.isOwner).toBe(true);
  });

  it("has nothing to fall back to on a first load that fails: the rail collapses", async () => {
    jest.mocked(getDatasetAccess).mockRejectedValue(new Error("boom"));
    jest.mocked(getGroupOverview).mockResolvedValue(GROUPS);

    const { result } = renderHook(() => useDatasetFacets("mine"));

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.accessById.size).toBe(0);
  });

  it("keeps the last good facets and access when a later refresh fails, and says so instead of going silent", async () => {
    jest.mocked(getDatasetAccess).mockResolvedValueOnce(ACCESS).mockRejectedValueOnce(new Error("boom"));
    jest.mocked(getGroupOverview).mockResolvedValue(GROUPS);

    const { result } = renderHook(() => useDatasetFacets("mine"));
    await waitFor(() => expect(result.current.status).toBe("ready"));

    await act(async () => { await result.current.refresh(); });

    // Still ready, with the access it already had, not collapsed to "error".
    expect(result.current.status).toBe("ready");
    expect(result.current.accessById.get("mine")?.isOwner).toBe(true);
    expect(notifications.show).toHaveBeenCalledWith(expect.objectContaining({ title: "Couldn't refresh who has access" }));
  });
});

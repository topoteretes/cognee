const mockGetMyUserId = jest.fn();
jest.mock("@/modules/apiKeys/getMyUserId", () => ({
  __esModule: true,
  default: () => mockGetMyUserId(),
}));

import getPodUserId from "@/modules/users/getPodUserId";
import type { CogneeInstance } from "@/modules/instances/types";

let counter = 0;

function instance(): CogneeInstance {
  counter += 1;
  return { name: "CloudCognee", instanceId: `instance-${counter}`, fetch: jest.fn() } as unknown as CogneeInstance;
}

describe("getPodUserId", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  // The control plane's id is the pod's id: the tenant sync inserts
  // User(id=user_id) with the id the control plane sent.
  it("uses the id the control plane already holds", async () => {
    mockGetMyUserId.mockResolvedValue("pod-user-1");

    expect(await getPodUserId(instance())).toBe("pod-user-1");
  });

  it("asks the pod for nothing", async () => {
    mockGetMyUserId.mockResolvedValue("pod-user-1");
    const podInstance = instance();

    await getPodUserId(podInstance);

    expect(podInstance.fetch).not.toHaveBeenCalled();
  });

  it("resolves once per instance and reuses the answer", async () => {
    mockGetMyUserId.mockResolvedValue("pod-user-1");
    const podInstance = instance();

    await getPodUserId(podInstance);
    await getPodUserId(podInstance);

    expect(mockGetMyUserId).toHaveBeenCalledTimes(1);
  });

  it("gives up when the control plane cannot say who is signed in", async () => {
    mockGetMyUserId.mockResolvedValue(null);

    expect(await getPodUserId(instance())).toBeNull();
  });

  it("gives up rather than throwing when the lookup fails", async () => {
    mockGetMyUserId.mockRejectedValue(new Error("network down"));

    expect(await getPodUserId(instance())).toBeNull();
  });

  it("retries after a failure rather than remembering it", async () => {
    mockGetMyUserId.mockResolvedValueOnce(null).mockResolvedValueOnce("pod-user-1");
    const podInstance = instance();

    expect(await getPodUserId(podInstance)).toBeNull();
    expect(await getPodUserId(podInstance)).toBe("pod-user-1");
  });
});

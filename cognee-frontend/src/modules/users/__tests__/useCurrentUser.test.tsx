import type { ReactNode } from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const mockGetCurrentUserProfile = jest.fn();
const mockIsCloudEnvironment = jest.fn();
jest.mock("@/modules/users/getCurrentUserProfile", () => ({
  __esModule: true,
  default: () => mockGetCurrentUserProfile(),
}));
jest.mock("@/utils/isCloudEnvironment", () => ({
  __esModule: true,
  default: () => mockIsCloudEnvironment(),
}));

import { useCurrentUser } from "../useCurrentUser";

function Wrapper({ children }: { children: ReactNode }): ReactNode {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useCurrentUser", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetCurrentUserProfile.mockResolvedValue(null);
    mockIsCloudEnvironment.mockReturnValue(true);
  });

  it("settles with no profile when the background profile read has no session", async () => {
    const { result } = renderHook(() => useCurrentUser(), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toBeNull();
    expect(result.current.isStale).toBe(true);
    expect(mockGetCurrentUserProfile).toHaveBeenCalledTimes(1);
  });

  it("does not read the cloud profile in local mode", () => {
    mockIsCloudEnvironment.mockReturnValue(false);

    const { result } = renderHook(() => useCurrentUser(), { wrapper: Wrapper });

    expect(result.current.fetchStatus).toBe("idle");
    expect(mockGetCurrentUserProfile).not.toHaveBeenCalled();
  });
});

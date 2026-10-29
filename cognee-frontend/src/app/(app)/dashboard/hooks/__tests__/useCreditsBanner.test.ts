import { renderHook, act } from "@testing-library/react";

const mockUseTenant = jest.fn();
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useTenant: () => mockUseTenant(),
}));

const mockUseCreditsBalance = jest.fn();
jest.mock("@/modules/billing/useCreditsBalance", () => ({
  useCreditsBalance: () => mockUseCreditsBalance(),
  getTenantRemainingUsd: jest.requireActual("@/modules/billing/useCreditsBalance").getTenantRemainingUsd,
}));

import { useCreditsBanner } from "../useCreditsBanner";

const TENANT = { tenant_id: "t1" };

// react-query hands back whatever the server action returned, and that is now
// the discriminated result rather than the bare overview — so the fixture wraps
// it. A raw overview here silently reads as "not ok" and every balance comes
// back null.
function balanceResult(row: Partial<{ spentUsd: number | null; maxBudgetUsd: number | null; remainingUsd: number | null }>) {
  return {
    status: "ok" as const,
    data: {
      budget: { totalUsd: 0, spentUsd: null, remainingUsd: null, spentPct: null },
      tenants: [{ tenantId: "t1", tenantName: "Personal", spentUsd: null, maxBudgetUsd: null, remainingUsd: null, ...row }],
      purchases: [],
    },
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  sessionStorage.clear();
  mockUseTenant.mockReturnValue({ tenant: TENANT });
});

describe("useCreditsBanner", () => {
  it("shows the low-balance banner when remaining balance is under $5", () => {
    mockUseCreditsBalance.mockReturnValue({ data: balanceResult({ remainingUsd: 2 }) });

    const { result } = renderHook(() => useCreditsBanner());

    expect(result.current.showLowBalanceBanner).toBe(true);
    expect(result.current.showVoucherBanner).toBe(false);
  });

  it("does not show the low-balance banner when remaining balance is $5 or more", () => {
    mockUseCreditsBalance.mockReturnValue({ data: balanceResult({ remainingUsd: 5 }) });

    const { result } = renderHook(() => useCreditsBanner());

    expect(result.current.showLowBalanceBanner).toBe(false);
    expect(result.current.showVoucherBanner).toBe(true);
  });

  it("derives remainingUsd from maxBudgetUsd - spentUsd when the API omits it directly", () => {
    mockUseCreditsBalance.mockReturnValue({ data: balanceResult({ remainingUsd: null, spentUsd: 96, maxBudgetUsd: 100 }) });

    const { result } = renderHook(() => useCreditsBanner());

    expect(result.current.creditsRemainingUsd).toBe(4);
    expect(result.current.showLowBalanceBanner).toBe(true);
  });

  it("reports no balance when remainingUsd, spentUsd and maxBudgetUsd are all null", () => {
    mockUseCreditsBalance.mockReturnValue({ data: balanceResult({}) });

    const { result } = renderHook(() => useCreditsBanner());

    expect(result.current.creditsRemainingUsd).toBeNull();
    expect(result.current.showLowBalanceBanner).toBe(false);
    expect(result.current.showVoucherBanner).toBe(true);
  });

  it("hides the low-balance banner for the session once dismissed", () => {
    mockUseCreditsBalance.mockReturnValue({ data: balanceResult({ remainingUsd: 0.5 }) });

    const { result } = renderHook(() => useCreditsBanner());
    expect(result.current.showLowBalanceBanner).toBe(true);

    act(() => {
      result.current.dismiss();
    });

    expect(result.current.showLowBalanceBanner).toBe(false);
  });

  // The point of the discriminated result, seen from a consumer: a failed
  // request must read as "no balance", never as a balance of zero. Treating it
  // as zero is what put a confident low-balance warning in front of users whose
  // request simply did not come back.
  it("reports no balance when the request failed, rather than a low one", () => {
    mockUseCreditsBalance.mockReturnValue({
      data: { status: "error" as const, message: "Server error (status 500)." },
    });

    const { result } = renderHook(() => useCreditsBanner());

    expect(result.current.creditsRemainingUsd).toBeNull();
    expect(result.current.showLowBalanceBanner).toBe(false);
  });

  it("keeps the voucher banner reachable off the raw balance, not the dismissible flag", () => {
    mockUseCreditsBalance.mockReturnValue({ data: balanceResult({ remainingUsd: 0.5 }) });

    const { result } = renderHook(() => useCreditsBanner());

    act(() => {
      result.current.dismiss();
    });

    expect(result.current.showLowBalanceBanner).toBe(false);
    expect(result.current.showVoucherBanner).toBe(false);
  });
});

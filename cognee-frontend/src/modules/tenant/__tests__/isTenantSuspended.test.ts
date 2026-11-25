import isTenantSuspended from "../isTenantSuspended";

const TENANTS = [
  { id: "awake", status: "active" },
  { id: "asleep", status: "suspended" },
  { id: "no-status" },
];

describe("isTenantSuspended", () => {
  it("is true for a tenant the control plane reports as suspended", () => {
    expect(isTenantSuspended(TENANTS, "asleep")).toBe(true);
  });

  it("is false for an active tenant", () => {
    expect(isTenantSuspended(TENANTS, "awake")).toBe(false);
  });

  it("is false when the backend sends no status field at all", () => {
    expect(isTenantSuspended(TENANTS, "no-status")).toBe(false);
  });

  it("is false for a tenant missing from the list", () => {
    expect(isTenantSuspended(TENANTS, "unknown-id")).toBe(false);
  });

  it("is false for an empty list", () => {
    expect(isTenantSuspended([], "asleep")).toBe(false);
  });

  it("is false for a status value it does not recognize", () => {
    expect(isTenantSuspended([{ id: "t", status: "provisioning" }], "t")).toBe(false);
  });
});

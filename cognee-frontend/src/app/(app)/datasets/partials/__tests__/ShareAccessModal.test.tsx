import { MantineProvider } from "@mantine/core";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import ShareAccessModal from "@/app/(app)/datasets/partials/ShareAccessModal";
import changeDatasetAccess from "@/modules/permissions/changeDatasetAccess";
import type { DatasetAccess } from "@/modules/permissions/types";
import type { ShareCandidate } from "@/app/(app)/datasets/partials/useShareCandidates";

jest.mock("@/modules/permissions/changeDatasetAccess");
jest.mock("@mantine/notifications", () => ({ notifications: { show: jest.fn() } }));
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: { fetch: jest.fn() } }),
}));

const CANDIDATES: ShareCandidate[] = [
  { id: "t-1", kind: "tenant", label: "Everyone in the workspace" },
  { id: "g-eng", kind: "role", label: "Engineering" },
  { id: "me", kind: "user", label: "me@acme.io" },
  { id: "zoe", kind: "user", label: "zoe@acme.io" },
];
jest.mock("@/app/(app)/datasets/partials/useShareCandidates", () => ({
  __esModule: true,
  WORKSPACE_LABEL: "Everyone in the workspace",
  default: () => CANDIDATES,
}));

const change = jest.mocked(changeDatasetAccess);

function access(sharedWith: DatasetAccess["sharedWith"]): DatasetAccess {
  return {
    datasetId: "ds-1", datasetName: "notes", ownerId: "me", isOwner: true,
    permissions: ["read", "write", "delete", "share"], sources: [], sharedWith,
  };
}

const OWNER = { principalType: "user" as const, principalId: "me", name: "me@acme.io", permissions: ["read", "write", "delete", "share"] as const, parentUserId: null };

function open(sharedWith: DatasetAccess["sharedWith"], onChanged = jest.fn()): jest.Mock {
  render(
    <MantineProvider>
      <ShareAccessModal datasetId="ds-1" datasetName="notes" access={access(sharedWith)} tenantId="t-1" onClose={jest.fn()} onChanged={onChanged} />
    </MantineProvider>,
  );
  return onChanged;
}

afterEach(() => {
  jest.resetAllMocks();
});

describe("ShareAccessModal", () => {
  it("says a dataset nobody else has is only the owner's", () => {
    open([{ ...OWNER, permissions: [...OWNER.permissions] }]);

    expect(screen.getByText("Only you can open this dataset.")).toBeInTheDocument();
  });

  it("lists who has it at which level, without the owner among them", () => {
    open([
      { ...OWNER, permissions: [...OWNER.permissions] },
      { principalType: "user", principalId: "zoe", name: "zoe@acme.io", permissions: ["read", "write"], parentUserId: null },
      { principalType: "tenant", principalId: "t-1", name: "Acme", permissions: ["read"], parentUserId: null },
    ]);

    expect(screen.getByRole("combobox", { name: "Access for zoe@acme.io" })).toHaveValue("edit");
    expect(screen.getByRole("combobox", { name: "Access for Everyone in the workspace" })).toHaveValue("view");
    expect(screen.queryByRole("combobox", { name: "Access for me@acme.io" })).not.toBeInTheDocument();
  });

  it("shares with a person at the chosen level, and shows it before the pod has answered", async () => {
    let answer: () => void = () => undefined;
    change.mockReturnValue(new Promise<void>((resolve) => { answer = resolve; }));
    const onChanged = open([{ ...OWNER, permissions: [...OWNER.permissions] }]);

    fireEvent.change(screen.getByRole("combobox", { name: "Share with" }), { target: { value: "zoe" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Level for the new access" }), { target: { value: "edit" } });
    fireEvent.click(screen.getByRole("button", { name: "Share" }));

    expect(screen.getByRole("combobox", { name: "Access for zoe@acme.io" })).toHaveValue("edit");
    expect(change).toHaveBeenCalledWith("zoe", "ds-1", "edit", expect.anything());
    expect(onChanged).not.toHaveBeenCalled();

    await act(async () => { answer(); });

    expect(onChanged).toHaveBeenCalledTimes(1);
  });

  it("can share with a group or with everyone, but never offers the owner or someone already added", () => {
    open([
      { ...OWNER, permissions: [...OWNER.permissions] },
      { principalType: "user", principalId: "zoe", name: "zoe@acme.io", permissions: ["read"], parentUserId: null },
    ]);

    const offered = [...screen.getByRole("combobox", { name: "Share with" }).querySelectorAll("option")].map((o) => o.textContent);

    expect(offered).toEqual(["Add a person, group or everyone…", "Everyone in the workspace", "Group: Engineering"]);
  });

  it("takes someone's access away", async () => {
    change.mockResolvedValue(undefined);
    open([
      { ...OWNER, permissions: [...OWNER.permissions] },
      { principalType: "tenant", principalId: "t-1", name: "Acme", permissions: ["read", "write"], parentUserId: null },
    ]);

    fireEvent.click(screen.getByRole("button", { name: "Remove Everyone in the workspace" }));

    expect(change).toHaveBeenCalledWith("t-1", "ds-1", null, expect.anything());
    await waitFor(() => expect(screen.getByText("Only you can open this dataset.")).toBeInTheDocument());
  });

  it("puts the list back to what the pod holds when it refuses", async () => {
    change.mockRejectedValue(new Error("not allowed"));
    open([
      { ...OWNER, permissions: [...OWNER.permissions] },
      { principalType: "user", principalId: "zoe", name: "zoe@acme.io", permissions: ["read"], parentUserId: null },
    ]);

    fireEvent.click(screen.getByRole("button", { name: "Remove zoe@acme.io" }));

    await waitFor(() => expect(screen.getByRole("combobox", { name: "Access for zoe@acme.io" })).toBeInTheDocument());
  });
});

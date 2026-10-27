import { ALL_FACET_ID, PERSONAL_FACET_ID, buildFacets, filterByFacet, withUnclaimedAsPersonal } from "@/app/(app)/datasets/partials/datasetFacets";
import type { DatasetAccess } from "@/modules/permissions/types";

function access(datasetId: string, overrides: Partial<DatasetAccess>): DatasetAccess {
  return {
    datasetId,
    datasetName: datasetId,
    ownerId: null,
    isOwner: false,
    permissions: ["read"],
    sources: [],
    sharedWith: null,
    ...overrides,
  };
}

const MINE = access("mine", {
  isOwner: true,
  sources: [{ principalType: "user", principalId: "me", name: "me@acme.io", permissions: ["read"], parentUserId: null }],
});
const VIA_ENGINEERING = access("eng-docs", {
  sources: [{ principalType: "role", principalId: "g-eng", name: "Engineering", permissions: ["read"], parentUserId: null }],
});
const SHARED_AND_GROUP = access("crm", {
  sources: [
    { principalType: "user", principalId: "me", name: "me@acme.io", permissions: ["read"], parentUserId: null },
    { principalType: "role", principalId: "g-eng", name: "Engineering", permissions: ["read"], parentUserId: null },
  ],
});
const WHOLE_WORKSPACE = access("handbook", {
  sources: [{ principalType: "tenant", principalId: "t-1", name: "Acme", permissions: ["read"], parentUserId: null }],
});

function idsOf(label: string, facets: ReturnType<typeof buildFacets>): string[] {
  return [...(facets.find((facet) => facet.label === label)?.datasetIds ?? [])].sort();
}

describe("datasetFacets", () => {
  const facets = buildFacets([MINE, VIA_ENGINEERING, SHARED_AND_GROUP, WHOLE_WORKSPACE], [
    { id: "g-eng", name: "Engineering", memberCount: 8 },
  ]);

  it("lists all, personal and shared before one entry per group", () => {
    expect(facets.map((facet) => facet.label)).toEqual(["All datasets", "Personal", "Shared with me", "Engineering"]);
  });

  it("puts only my own datasets under Personal", () => {
    expect(idsOf("Personal", facets)).toEqual(["mine"]);
  });

  it("counts a personal share and a workspace-wide grant as shared with me", () => {
    expect(idsOf("Shared with me", facets)).toEqual(["crm", "handbook"]);
  });

  it("shows a dataset reachable two ways under both, as one dataset each time", () => {
    expect(idsOf("Engineering", facets)).toEqual(["crm", "eng-docs"]);
  });

  it("describes a group by its member count", () => {
    expect(facets.find((facet) => facet.label === "Engineering")?.hint).toBe("8 members");
  });

  it("keeps a just-created dataset under All before the access list knows it", () => {
    const all = facets.find((facet) => facet.id === ALL_FACET_ID);
    expect(filterByFacet([{ id: "brand-new" }, { id: "mine" }], all)).toEqual([{ id: "brand-new" }, { id: "mine" }]);
  });

  it("filters to the facet's datasets otherwise", () => {
    const personal = facets.find((facet) => facet.label === "Personal");
    expect(filterByFacet([{ id: "brand-new" }, { id: "mine" }], personal)).toEqual([{ id: "mine" }]);
  });
});

describe("a group I am in that holds nothing yet", () => {
  const groups = [
    { id: "g-new", name: "Design", memberCount: 3 },
    { id: "g-other", name: "Finance", memberCount: 2 },
  ];

  it("is listed, and says so", () => {
    const built = buildFacets([MINE], groups, new Set(["g-new"]));

    const design = built.find((facet) => facet.label === "Design");
    expect(design?.hint).toBe("3 members · no datasets yet");
    expect(design?.datasetIds.size).toBe(0);
  });

  it("is the only one listed: a group I am not in stays out", () => {
    const built = buildFacets([MINE], groups, new Set(["g-new"]));

    expect(built.map((facet) => facet.label)).toEqual(["All datasets", "Personal", "Shared with me", "Design"]);
  });

  it("is not duplicated when it does hold a dataset", () => {
    const built = buildFacets([VIA_ENGINEERING], [{ id: "g-eng", name: "Engineering", memberCount: 8 }], new Set(["g-eng"]));

    expect(built.filter((facet) => facet.label === "Engineering")).toHaveLength(1);
    expect(built.find((facet) => facet.label === "Engineering")?.hint).toBe("8 members");
  });
});

describe("two or more groups in the rail", () => {
  it("sorts them alphabetically by name, not by when they were granted access", () => {
    const built = buildFacets(
      [
        access("wiki", { sources: [{ principalType: "role", principalId: "g-z", name: "Zebra", permissions: ["read"], parentUserId: null }] }),
        access("crm", { sources: [{ principalType: "role", principalId: "g-a", name: "Acme Sales", permissions: ["read"], parentUserId: null }] }),
      ],
      [],
    );

    expect(built.filter((facet) => facet.kind === "group").map((facet) => facet.label)).toEqual(["Acme Sales", "Zebra"]);
  });

  it("says one member in the singular and anything else in the plural", () => {
    const built = buildFacets([VIA_ENGINEERING], [{ id: "g-eng", name: "Engineering", memberCount: 1 }]);

    expect(built.find((facet) => facet.label === "Engineering")?.hint).toBe("1 member");
  });

  it("falls back to a generic label when the group's member count is unknown", () => {
    const built = buildFacets([VIA_ENGINEERING], [{ id: "g-eng", name: "Engineering", memberCount: undefined as unknown as number }]);

    expect(built.find((facet) => facet.label === "Engineering")?.hint).toBe("Group");
  });
});

describe("filterByFacet with no facet selected", () => {
  it("shows everything, the same as the All facet would", () => {
    expect(filterByFacet([{ id: "a" }, { id: "b" }], undefined)).toEqual([{ id: "a" }, { id: "b" }]);
  });
});

describe("withUnclaimedAsPersonal", () => {
  const facets = buildFacets([MINE, VIA_ENGINEERING], [{ id: "g-eng", name: "Engineering", memberCount: 8 }]);

  it("adds an id this session knows is its own to Personal", () => {
    const built = withUnclaimedAsPersonal(facets, ["pending-123"]);

    const personal = built.find((facet) => facet.id === PERSONAL_FACET_ID);
    expect([...(personal?.datasetIds ?? [])].sort()).toEqual(["mine", "pending-123"]);
  });

  it("leaves every other facet alone", () => {
    const built = withUnclaimedAsPersonal(facets, ["pending-123"]);

    const engineering = built.find((facet) => facet.label === "Engineering");
    expect(engineering?.datasetIds.has("pending-123")).toBe(false);
  });

  it("changes nothing when there is nothing new to fold in", () => {
    const built = withUnclaimedAsPersonal(facets, []);

    expect(built).toEqual(facets);
  });
});

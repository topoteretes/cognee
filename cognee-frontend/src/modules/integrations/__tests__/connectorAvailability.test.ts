import { absentEntry, isOffered } from "../connectorAvailability";
import type { ConnectorCatalogEntry } from "../connectorTypes";
import type { TeamConnectorCfg } from "../types";

const DRIVE: TeamConnectorCfg = {
  key: "google_drive",
  name: "Google Drive",
  description: "Folders",
  initials: "Dr",
  color: "#4285F4",
  permissions: [],
  supportsChannelRouting: false,
  api: "framework",
  resourceNoun: { singular: "folder", plural: "folders" },
  selectionNote: "Cognee reads the folders selected below.",
};

const SLACK: TeamConnectorCfg = { ...DRIVE, key: "slack", name: "Slack", api: "slack" };

const GITHUB_ENTRY: ConnectorCatalogEntry = {
  provider: "github",
  displayName: "GitHub",
  configured: true,
  supportsResourceSelection: true,
  degraded: false,
};

describe("isOffered", () => {
  it("is not yet known while the catalog is still being read", () => {
    expect(isOffered(DRIVE, undefined)).toBeUndefined();
  });

  it("offers everything after a failed read, so a failed read can never hide a live connector", () => {
    expect(isOffered(DRIVE, null)).toBe(true);
  });

  it("says no for a framework provider a read catalog does not list", () => {
    expect(isOffered(DRIVE, [GITHUB_ENTRY])).toBe(false);
  });

  it("says no on a real empty catalog too — that is an answer, not a failure", () => {
    expect(isOffered(DRIVE, [])).toBe(false);
  });

  it("offers a listed provider whether or not it is configured, since an old install still manages", () => {
    expect(isOffered(DRIVE, [{ ...GITHUB_ENTRY, provider: "google_drive", configured: false }])).toBe(true);
  });

  it("always offers Slack, which is off the framework and never in the catalog", () => {
    expect(isOffered(SLACK, undefined)).toBe(true);
    expect(isOffered(SLACK, [])).toBe(true);
  });
});

describe("absentEntry", () => {
  it("reads as unconfigured, so the card says 'not switched on here yet' and offers nothing", () => {
    expect(absentEntry(DRIVE)).toEqual({
      provider: "google_drive",
      displayName: "Google Drive",
      configured: false,
      supportsResourceSelection: false,
      degraded: false,
    });
  });
});

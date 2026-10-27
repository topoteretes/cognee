import { existsSync } from "node:fs";
import path from "node:path";
import { DATA_SOURCE_CARDS, isProviderShownOnHost } from "../dataSourceCards";

// src/modules/integrations/__tests__ → the frontend root, whatever jest's cwd is.
const LOGO_DIR = path.resolve(__dirname, "../../../../public/visuals/logos/datasources");

describe("DATA_SOURCE_CARDS", () => {
  it("lists Google Drive as a framework connector with a folder picker", () => {
    const drive = DATA_SOURCE_CARDS.find((card) => card.key === "google_drive");

    expect(drive).toMatchObject({
      api: "framework",
      resourceNoun: { singular: "folder", plural: "folders" },
      supportsDataDeletion: true,
    });
  });

  it("lists Google Calendar as a framework connector that needs an explicit selection and can delete its data", () => {
    const calendar = DATA_SOURCE_CARDS.find((card) => card.key === "google_calendar");

    expect(calendar).toMatchObject({
      api: "framework",
      resourceNoun: { singular: "calendar", plural: "calendars" },
      supportsChannelRouting: false,
      requiresSelection: true,
      supportsDataDeletion: true,
    });
  });

  it("draws Google Calendar with its own glyph, not another card's", () => {
    const calendar = DATA_SOURCE_CARDS.find((card) => card.key === "google_calendar");

    expect(calendar?.logo).toBe("google_calendar");
  });

  it("tells the owner that unticking a calendar forgets its events", () => {
    const calendar = DATA_SOURCE_CARDS.find((card) => card.key === "google_calendar");

    expect(calendar?.selectionNote).toMatch(/forgets its events/);
  });

  it("lists Linear as a framework connector with a team picker that can delete its data", () => {
    const linear = DATA_SOURCE_CARDS.find((card) => card.key === "linear");

    expect(linear).toMatchObject({
      api: "framework",
      resourceNoun: { singular: "team", plural: "teams" },
      supportsChannelRouting: false,
      supportsDataDeletion: true,
    });
  });

  it("lists Notion as a framework connector with a page picker that reconnects to add more", () => {
    const notion = DATA_SOURCE_CARDS.find((card) => card.key === "notion");

    expect(notion).toMatchObject({
      api: "framework",
      resourceNoun: { singular: "page", plural: "pages" },
      supportsChannelRouting: false,
      addResourcesViaReconnect: true,
    });
    // supports_shared_dataset_delete stays False on the backend until the pod
    // builder reports the tables it writes (CLO-497) — this must not offer a
    // delete the backend would refuse.
    expect(notion?.supportsDataDeletion).toBeUndefined();
  });

  it("keys every card uniquely — the key is the provider the backend routes on", () => {
    const keys = DATA_SOURCE_CARDS.map((card) => card.key);

    expect(new Set(keys).size).toBe(keys.length);
  });

  it("ships a glyph for every card that names one", () => {
    for (const card of DATA_SOURCE_CARDS) {
      if (card.logo === undefined) continue;
      expect(existsSync(path.join(LOGO_DIR, `${card.logo}.svg`))).toBe(true);
    }
  });
});

describe("isProviderShownOnHost (CLO-1100)", () => {
  it("lists Google Drive on dev, previews and local dev", () => {
    for (const host of ["dev.cognee.ai", "cognee-saas-git-x.vercel.app", "localhost"]) {
      expect(isProviderShownOnHost("google_drive", host)).toBe(true);
    }
  });

  it("hides Google Drive on staging, production, and before the host is known", () => {
    for (const host of ["staging.cognee.ai", "platform.cognee.ai", null]) {
      expect(isProviderShownOnHost("google_drive", host)).toBe(false);
    }
  });

  it("lists Google Calendar on dev, previews and local dev only, until its scope is verified", () => {
    for (const host of ["dev.cognee.ai", "cognee-saas-git-x.vercel.app", "localhost"]) {
      expect(isProviderShownOnHost("google_calendar", host)).toBe(true);
    }
    for (const host of ["staging.cognee.ai", "platform.cognee.ai", null]) {
      expect(isProviderShownOnHost("google_calendar", host)).toBe(false);
    }
  });

  it("lists every other connector everywhere", () => {
    expect(isProviderShownOnHost("slack", "platform.cognee.ai")).toBe(true);
    expect(isProviderShownOnHost("github", null)).toBe(true);
    expect(isProviderShownOnHost("linear", "platform.cognee.ai")).toBe(true);
    expect(isProviderShownOnHost("linear", null)).toBe(true);
  });

  it("lists Notion everywhere, staging and production included", () => {
    for (const host of ["dev.cognee.ai", "localhost", "staging.cognee.ai", "platform.cognee.ai", null]) {
      expect(isProviderShownOnHost("notion", host)).toBe(true);
    }
  });
});

import { fireEvent, render, screen } from "@testing-library/react";
import MoreDataSourcesSection from "../MoreDataSourcesSection";

const mockTrackEvent = jest.fn();
jest.mock("@/modules/analytics", () => ({
  trackEvent: (...args: unknown[]) => mockTrackEvent(...args),
}));

function cards(): HTMLElement[] {
  return screen.getAllByRole("button").filter((button) => button.classList.contains("ds-card"));
}

/** A card's name line: logo tile first, then name over description. */
function nameOf(card: HTMLElement): string {
  return card.children[1].children[0].textContent ?? "";
}

function cardNames(): string[] {
  return cards().map(nameOf);
}

describe("MoreDataSourcesSection", () => {
  beforeEach(() => mockTrackEvent.mockReset());

  it("sends every card's own name with its Connect click and its notify request", () => {
    render(<MoreDataSourcesSection />);

    for (const card of cards()) {
      const name = nameOf(card);
      fireEvent.click(card);
      expect(mockTrackEvent).toHaveBeenLastCalledWith({
        pageName: "Integrations",
        eventName: "data_source_connect_clicked",
        additionalProperties: { data_source: name },
      });

      fireEvent.click(screen.getByRole("button", { name: "Get notified once live" }));
      expect(mockTrackEvent).toHaveBeenLastCalledWith({
        pageName: "Integrations",
        eventName: "data_source_notify_requested",
        additionalProperties: { data_source: name },
      });

      fireEvent.click(screen.getByRole("button", { name: "Close" }));
    }
  });

  it("leaves out the connectors that are live above it, so their clicks never read as false-door demand", () => {
    render(<MoreDataSourcesSection />);

    const names = cardNames();
    for (const live of ["GitHub", "Google Drive", "Gmail", "Slack", "Linear", "Notion"]) expect(names).not.toContain(live);
  });

  it("offers the rest of the Atlassian suite, the Microsoft 365 apps, Salesforce and Obsidian", () => {
    render(<MoreDataSourcesSection />);

    expect(cardNames()).toEqual(
      expect.arrayContaining([
        "Trello",
        "Bitbucket",
        "Loom",
        "Jira Service Management",
        "SharePoint",
        "OneDrive",
        "Microsoft Teams",
        "Outlook",
        "OneNote",
        "Word",
        "Excel",
        "PowerPoint",
        "Obsidian",
        "Salesforce",
      ]),
    );
  });
});

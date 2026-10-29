import { render, screen } from "@testing-library/react";
import WorkspaceBadge from "../WorkspaceBadge";

const COLOR = "#6510F4";

function badgeFor(state: "active" | "asleep" | "waking" | "blocked", size = 16): HTMLElement {
  render(<WorkspaceBadge initial="R" color={COLOR} size={size} state={state} />);
  const letter = screen.getByText("R");
  const badge = letter.parentElement;
  if (!badge) throw new Error("badge element missing");
  return badge;
}

describe("WorkspaceBadge", () => {
  it("fills with the workspace colour while the workspace is up", () => {
    const badge = badgeFor("active");
    expect(badge.style.background).toBe("rgb(101, 16, 244)");
    expect(badge.style.borderStyle).toBe("");
  });

  it("goes hollow when the workspace is asleep, so a shut picker still shows it", () => {
    const badge = badgeFor("asleep");
    expect(badge.style.background).toBe("transparent");
    expect(badge.style.borderStyle).toBe("dashed");
  });

  it("lightens the workspace colour for the outline, which the raw palette is too dark for", () => {
    const badge = badgeFor("asleep");
    expect(badge.style.borderColor).toBe("color-mix(in srgb, rgb(101, 16, 244) 55%, rgb(237, 236, 234))");
  });

  it("stays hollow but animates while the workspace is being woken", () => {
    const badge = badgeFor("waking");
    expect(badge.style.borderStyle).toBe("dashed");
    // The literal class name, not just truthiness: next/jest maps the CSS
    // module through an identity proxy, so any property at all comes back
    // truthy and an assertion on that can never fail.
    expect(badge.className).toBe("waking");
  });

  it("does not animate in any other state", () => {
    expect(badgeFor("asleep").className).toBe("");
  });

  it("lightens the glyph too, which is the half that has to stay legible", () => {
    badgeFor("asleep");
    expect(screen.getByText("R").style.color).toBe("color-mix(in srgb, rgb(101, 16, 244) 55%, rgb(237, 236, 234))");
  });

  it("keeps a readable colour on the wrapper so a dropped color-mix cannot leave an invisible badge", () => {
    const badge = badgeFor("asleep");
    // A browser that cannot parse color-mix drops that declaration; without
    // this the letter falls back to the UA default, black on near-black chrome.
    expect(badge.style.color).toBe("rgb(237, 236, 234)");
    expect(badge.style.borderStyle).toBe("dashed");
  });

  it("names a wake for a screen reader, which a dashed border cannot", () => {
    expect(badgeFor("waking").getAttribute("aria-label")).toBe("R workspace, waking up");
  });

  it("names a sleeping workspace for a screen reader", () => {
    expect(badgeFor("asleep").getAttribute("aria-label")).toBe("R workspace, asleep");
  });

  it("scales the corner and the glyph with the top bar's larger badge", () => {
    const badge = badgeFor("active", 20);
    expect(badge.style.borderRadius).toBe("4px");
    expect(screen.getByText("R").style.fontSize).toBe("10px");
  });

  it("greys out a workspace with no subscription instead of hollowing it", () => {
    const badge = badgeFor("blocked");
    expect(badge.style.background).toBe("rgb(85, 85, 85)");
    expect(badge.style.borderStyle).toBe("");
  });
});

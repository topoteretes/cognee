import React from "react";
import { render } from "@testing-library/react";
import { AsciiFrame } from "../AsciiFrame";

describe("AsciiFrame body sizing", () => {
  it("lets the body shrink below its content, so a scrolling body can scroll", () => {
    // The regression this guards: as a flex item the body div defaults to
    // `min-height: auto` and refuses to shrink below its content. A caller
    // that constrains the frame's height and puts a scroll box inside got a
    // body that grew to its content instead, the section's `overflow: hidden`
    // clipped the excess, and the scroll box never scrolled because it had
    // been handed all the height it asked for. Measured on the activity
    // table: the body stood at 1686px inside a 647px section.
    //
    // jsdom measures no layout, so this pins the declaration rather than the
    // outcome. It is the whole fix, and it is one property.
    const { container } = render(
      <AsciiFrame label={null} style={{ flex: 1, minHeight: 0 }}>
        <div>body</div>
      </AsciiFrame>,
    );

    const section = container.querySelector("section");
    const body = section?.querySelector(":scope > div");
    expect(body?.getAttribute("style")).toContain("min-height: 0");
    expect(body?.getAttribute("style")).toContain("flex: 1");
  });

  it("still passes a caller's style through to the frame itself", () => {
    // How the activity page makes its own frame height-definite without
    // touching the other surfaces that use this component.
    const { container } = render(
      <AsciiFrame label={null} style={{ flex: 1, minHeight: 0 }}>
        <div>body</div>
      </AsciiFrame>,
    );

    expect(container.querySelector("section")?.getAttribute("style")).toContain("min-height: 0");
  });

  it("keeps the header out of the flexible area when there is a label", () => {
    // The header is flexShrink: 0 so the body absorbs the height instead.
    const { container } = render(<AsciiFrame label="Panel"><div>body</div></AsciiFrame>);
    expect(container.querySelector("header")?.getAttribute("style")).toContain("flex-shrink: 0");
  });
});

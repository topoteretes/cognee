import { render, screen } from "@testing-library/react";
import { MemoryBrain } from "../MemoryBrain";
import { brainLayout } from "../brainLayout";
import { FULL, type BrainAmounts } from "../brainSections";

const EMPTY: BrainAmounts = {
  teams: 0,
  dataSources: 0,
  integrations: 0,
  sessions: 0,
  agents: 0,
  connections: 0,
};
const TOTAL = brainLayout().cells.length;

const rects = (container: HTMLElement, selector: string) =>
  container.querySelectorAll(`rect${selector}`);

describe("MemoryBrain", () => {
  describe("with nothing in it", () => {
    it("draws the empty brain, every square waiting", () => {
      const { container } = render(<MemoryBrain amounts={EMPTY} />);

      expect(
        screen.getByRole("img", {
          name: `Your memory: 0 of ${TOTAL.toLocaleString("en-US")} squares filled`,
        }),
      ).toBeInTheDocument();
      expect(rects(container, '[class*="fill-brain-lavender"]')).toHaveLength(0);
      expect(rects(container, ".fill-cognee-fg\\/14")).toHaveLength(TOTAL);
    });
  });

  describe("part way", () => {
    const amounts: BrainAmounts = { ...EMPTY, dataSources: 5, agents: 2 };

    it("fills a lobe in its own shade and leaves the others empty", () => {
      const { container } = render(<MemoryBrain amounts={amounts} />);

      expect(rects(container, ".fill-brain-lavender-4").length).toBeGreaterThan(0);
      expect(rects(container, ".fill-brain-lavender-3").length).toBeGreaterThan(0);
      expect(rects(container, ".fill-brain-lavender-1")).toHaveLength(0);
    });

    it("fills a lobe more the more it holds", () => {
      const few = render(<MemoryBrain amounts={{ ...EMPTY, dataSources: 2 }} />);
      const fewLit = rects(few.container, ".fill-brain-lavender-4").length;
      few.unmount();
      const many = render(<MemoryBrain amounts={{ ...EMPTY, dataSources: 8 }} />);

      expect(rects(many.container, ".fill-brain-lavender-4").length).toBeGreaterThan(fewLit);
    });

    it("says how far each lobe has filled when it is hovered", () => {
      const { container } = render(<MemoryBrain amounts={amounts} />);

      const titles = [...container.querySelectorAll("title")].map((title) => title.textContent);
      expect(titles).toContain("Data sources: 5 of 10");
      expect(titles).toContain("Agents: 2 of 10");
      expect(titles).toContain("Connections: 0 of 40");
    });

    it("reads a part of an item with at most one decimal, and a whole one without", () => {
      const { container } = render(<MemoryBrain amounts={{ ...amounts, dataSources: 1.4738560627359831, agents: 2.04, sessions: 3.5 }} />);

      const titles = [...container.querySelectorAll("title")].map((title) => title.textContent);
      expect(titles).toContain("Data sources: 1.5 of 10");
      expect(titles).toContain("Agents: 2 of 10");
      expect(titles).toContain("Sessions: 3.5 of 10");
    });
  });

  describe("complete", () => {
    it("fills every square once every lobe is full", () => {
      render(<MemoryBrain amounts={FULL} />);

      const total = TOTAL.toLocaleString("en-US");
      expect(
        screen.getByRole("img", { name: `Your memory: ${total} of ${total} squares filled` }),
      ).toBeInTheDocument();
    });

    it("stays full for more than full", () => {
      render(<MemoryBrain amounts={{ ...FULL, agents: FULL.agents * 3 }} />);

      const total = TOTAL.toLocaleString("en-US");
      expect(
        screen.getByRole("img", { name: `Your memory: ${total} of ${total} squares filled` }),
      ).toBeInTheDocument();
    });
  });

  describe("while it is still being read in", () => {
    it("blinks only the squares that are filled", () => {
      const { container } = render(
        <MemoryBrain amounts={{ ...EMPTY, dataSources: 5 }} filling />,
      );

      const blinking = rects(container, ".motion-safe\\:animate-pulse").length;
      expect(blinking).toBe(rects(container, ".fill-brain-lavender-4").length);
      expect(blinking).toBeGreaterThan(0);
    });

    it("does not blink when it is not filling", () => {
      const { container } = render(<MemoryBrain amounts={{ ...EMPTY, dataSources: 5 }} />);

      expect(rects(container, ".motion-safe\\:animate-pulse")).toHaveLength(0);
    });
  });

  it("carries no inline style", () => {
    const { container } = render(<MemoryBrain amounts={FULL} />);

    expect(container.querySelector("[style]")).toBeNull();
  });
});

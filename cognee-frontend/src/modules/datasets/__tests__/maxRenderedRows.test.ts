import { capRows, MAX_RENDERED_ROWS } from "../maxRenderedRows";

const rows = (n: number) => Array.from({ length: n }, (_, i) => ({ id: String(i) }));

describe("capRows", () => {
  it("passes a short list through untouched", () => {
    const input = rows(12);
    const { visible, hidden } = capRows(input);

    expect(visible).toBe(input);
    expect(hidden).toBe(0);
  });

  it("passes a list exactly at the cap through untouched", () => {
    const { visible, hidden } = capRows(rows(MAX_RENDERED_ROWS));

    expect(visible).toHaveLength(MAX_RENDERED_ROWS);
    expect(hidden).toBe(0);
  });

  it("bounds a list one over the cap", () => {
    const { visible, hidden } = capRows(rows(MAX_RENDERED_ROWS + 1));

    expect(visible).toHaveLength(MAX_RENDERED_ROWS);
    expect(hidden).toBe(1);
  });

  it("bounds the dataset size that froze the tab, and reports the true total", () => {
    const total = 171_828;
    const { visible, hidden } = capRows(rows(total));

    expect(visible).toHaveLength(MAX_RENDERED_ROWS);
    expect(visible.length + hidden).toBe(total);
  });

  it("keeps the leading rows, in order", () => {
    const { visible } = capRows(rows(MAX_RENDERED_ROWS + 50));

    expect(visible[0].id).toBe("0");
    expect(visible[visible.length - 1].id).toBe(String(MAX_RENDERED_ROWS - 1));
  });

  it("handles an empty list", () => {
    expect(capRows([])).toEqual({ visible: [], hidden: 0 });
  });
});

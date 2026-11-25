import { render, screen } from "@testing-library/react";
import StructuredRowDetails from "@/modules/business/panels/StructuredRowDetails";

const header = "Table: people Columns:\n  - person (TEXT)";

describe("StructuredRowDetails", () => {
  it("shows nothing for a compact row whose values were never read", () => {
    const { container } = render(<StructuredRowDetails node={{ id: "r", type: "DltRow", name: header }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows nothing for a row marked as asked with no values, instead of its schema header", () => {
    const { container } = render(<StructuredRowDetails node={{ id: "r", type: "DltRow", name: header, text: "" }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("lists the row's columns and values when they are available", () => {
    render(<StructuredRowDetails node={{ id: "r", type: "DltRow", text: "Row Data:\n  person: Ana" }} />);
    expect(screen.getByRole("table", { name: "Row values" })).toHaveTextContent("Ana");
  });
});

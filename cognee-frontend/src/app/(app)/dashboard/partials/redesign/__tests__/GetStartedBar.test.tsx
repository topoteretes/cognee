import { render, screen } from "@testing-library/react";
import { GetStartedBar } from "../GetStartedBar";

describe("GetStartedBar connector count", () => {
  it("shows the confirmed count once telemetry has answered", () => {
    render(<GetStartedBar subtitle="sub" connectors={3}>children</GetStartedBar>);

    expect(screen.getByText("3 connectors")).toBeInTheDocument();
  });

  it("shows a neutral placeholder instead of a confident zero while still loading", () => {
    // "0 connectors" while the first poll is still in flight is the false
    // claim this exists to prevent — it hasn't been confirmed yet.
    render(<GetStartedBar subtitle="sub" connectors={0} connectorsLoading>children</GetStartedBar>);

    expect(screen.getByText("Checking connectors…")).toBeInTheDocument();
    expect(screen.queryByText("0 connectors")).not.toBeInTheDocument();
  });

  it("defaults to showing the count when connectorsLoading is not passed at all", () => {
    render(<GetStartedBar subtitle="sub" connectors={1}>children</GetStartedBar>);

    expect(screen.getByText("1 connector")).toBeInTheDocument();
  });
});

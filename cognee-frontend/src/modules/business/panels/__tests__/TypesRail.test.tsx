import { MantineProvider } from "@mantine/core";
import { render, screen } from "@testing-library/react";
import TypesRail from "../TypesRail";
import { createTypeEncoding } from "../../colorEncoding";
import type { BusinessEntity, TypeNode } from "../../sceneTypes";

jest.mock("@mantine/core", () => {
  const actual = jest.requireActual("@mantine/core");
  return {
    ...actual,
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  };
});

function typeNode(name: string, count: number): TypeNode {
  const members: BusinessEntity[] = Array.from({ length: count }, (_, i) => ({ id: `${name}-${i}` }));
  return { name, members, sets: { crm: count } };
}

describe("TypesRail", () => {
  it("lists the untyped bucket after the typed ones however big it is", () => {
    render(
      <MantineProvider>
        <TypesRail
          typeNodes={[typeNode("other", 50), typeNode("person", 3), typeNode("country", 5)]}
          encoding={createTypeEncoding({ person: "#111111", country: "#222222" }, () => false)}
          focusTypes={null}
          onToggleFocus={jest.fn()}
        />
      </MantineProvider>,
    );
    const names = screen.getAllByText(/^(other|person|country)$/).map((el) => el.textContent);
    expect(names).toEqual(["country", "person", "other"]);
  });

  it("renders nothing without type nodes", () => {
    render(
      <MantineProvider>
        <TypesRail typeNodes={null} encoding={createTypeEncoding({}, () => false)} focusTypes={null} onToggleFocus={jest.fn()} />
      </MantineProvider>,
    );
    expect(screen.queryByText("types")).toBeNull();
  });
});

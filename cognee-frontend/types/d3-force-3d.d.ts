declare module "d3-force-3d" {
  interface SimulationNodeDatum {
    index?: number;
    x?: number;
    y?: number;
    z?: number;
    vx?: number;
    vy?: number;
    vz?: number;
    fx?: number | null;
    fy?: number | null;
    fz?: number | null;
  }

  interface Force<Nodes extends SimulationNodeDatum = SimulationNodeDatum> {
    (alpha: number): void;
  }

  type ForceStrength<Nodes extends SimulationNodeDatum> =
    | number
    | ((node: Nodes, index: number, nodes: Nodes[]) => number);

  export function forceCollide<Nodes extends SimulationNodeDatum = SimulationNodeDatum>(
    radius?: number | ((node: Nodes, index: number, nodes: Nodes[]) => number)
  ): Force<Nodes> & {
    iterations(): number;
    iterations(iterations: number): this;
    radius(): (node: Nodes, index: number, nodes: Nodes[]) => number;
    radius(radius: number | ((node: Nodes, index: number, nodes: Nodes[]) => number)): this;
  };

  export function forceManyBody<Nodes extends SimulationNodeDatum = SimulationNodeDatum>(): Force<Nodes> & {
    strength(): (node: Nodes, index: number, nodes: Nodes[]) => number;
    strength(strength: ForceStrength<Nodes>): this;
    theta(): number;
    theta(theta: number): this;
    distanceMin(): number;
    distanceMin(distance: number): this;
    distanceMax(): number;
    distanceMax(distance: number): this;
  };
}

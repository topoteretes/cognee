declare module "d3-force-3d" {
  // Import types from d3-force if needed
  import {
    SimulationNodeDatum,
    SimulationLinkDatum,
    Force,
    Simulation,
  } from "d3-force";

  export interface SimulationNodeDatum3D extends SimulationNodeDatum {
    x: number;
    y: number;
    z: number;
    vx: number;
    vy: number;
    vz: number;
    fx?: number | null;
    fy?: number | null;
    fz?: number | null;
  }

  export function forceSimulation<NodeDatum extends SimulationNodeDatum3D>(
    nodes?: NodeDatum[]
  ): Simulation<NodeDatum, undefined>;

  export function forceCenter(x: number, y: number, z: number): Force<SimulationNodeDatum3D, any>;

  export function forceManyBody(): Force<SimulationNodeDatum3D, any>;

  export function forceLink<NodeDatum extends SimulationNodeDatum3D, Links extends SimulationLinkDatum<NodeDatum>[] = SimulationLinkDatum<NodeDatum>[]>(
    links?: Links
  ): Force<NodeDatum, SimulationLinkDatum<NodeDatum>>;

  export function forceCollide(radius?: number): Force<SimulationNodeDatum3D, any>;

  export function forceRadial(radius: number, x?: number, y?: number, z?: number): Force<SimulationNodeDatum3D, any>;

  export function forceX(x?: number): Force<SimulationNodeDatum3D, any>;
  export function forceY(y?: number): Force<SimulationNodeDatum3D, any>;
  export function forceZ(z?: number): Force<SimulationNodeDatum3D, any>;
}

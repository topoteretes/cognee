declare module "d3-force-3d" {
  // Minimal typing for the d3-force force-function protocol: a force is a
  // callable invoked each simulation tick, with chainable config setters.
  export interface Force {
    (alpha: number): void;
    initialize?(nodes: unknown[], random: () => number): void;
  }

  export interface ForceManyBody extends Force {
    strength(value: number): ForceManyBody;
    distanceMin(value: number): ForceManyBody;
    distanceMax(value: number): ForceManyBody;
  }

  export function forceCollide(radius?: number): Force;
  export function forceManyBody(): ForceManyBody;
}

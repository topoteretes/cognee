declare module "d3-force-3d" {
  // Minimal typing for the d3-force force-function protocol: a force is a
  // callable invoked each simulation tick, with chainable config setters.
  //
  // Open-source override — the public repo previously carried its own,
  // richer ambient declaration for this module here. Declaration merging
  // between that file and the shared shim shipped in
  // src/app/(graph)/d3-force-3d.d.ts made TypeScript pick the wrong
  // `forceManyBody`/`forceCollide` overload (the generic, 3D-node one),
  // breaking the 2D GraphVisualization call site. Only GraphVisualization.tsx
  // consumes this module in either repo, so this replaces the legacy
  // declaration with the same shim src/app/(graph)/d3-force-3d.d.ts uses.
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

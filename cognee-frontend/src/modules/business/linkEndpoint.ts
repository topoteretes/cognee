// d3 replaces a link's source/target ids with node objects once a simulation
// has run over it, so anything reading endpoints must accept both shapes.
export function endId(v: unknown): string {
  return typeof v === "object" && v !== null ? String((v as { id: string }).id) : String(v);
}

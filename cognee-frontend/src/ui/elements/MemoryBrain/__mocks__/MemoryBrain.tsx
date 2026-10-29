import type { ReactElement } from "react";
import type { BrainAmounts } from "../brainSections";

// A stand-in for tests of what holds a brain, not of the brain: the real one
// draws about ten thousand squares, and every query a test makes walks all of
// them, which on a slow runner is enough to miss a one second wait. It keeps the
// name and role the real one has, and counts what it was given, so a test can
// tell "nothing yet" from "something".
export function MemoryBrain({ amounts, filling = false }: { amounts: BrainAmounts; filling?: boolean }): ReactElement {
  const held = Object.values(amounts).reduce((sum, count) => sum + count, 0);
  return <svg role="img" data-filling={filling} aria-label={`Your memory: ${held} of 10,685 squares filled`} />;
}

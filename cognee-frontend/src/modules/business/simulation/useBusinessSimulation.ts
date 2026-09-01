"use client";

import { useEffect, useRef } from "react";
import { forceSimulation, forceLink, forceManyBody, forceCollide, forceX, forceY } from "d3-force";
import type { BusinessEntity, SemanticLink, Anchor } from "../sceneTypes";
import { setsOf } from "../computeBrainState";

const NEWBORN_BASE_DELAY_MS = 400;
const NEWBORN_STAGGER_MS = 40;
// A same-id-set restart (see idsKey below) only needs to re-seed cached
// positions onto the fresh object instances a refetch hands back — it has
// nothing new to settle, so it starts cold enough that d3-force's alphaMin
// cutoff stops it within a tick or two, rather than visibly re-animating.
const ALPHA_UNCHANGED_IDS = 0.02;
const ALPHA_CHANGED_IDS = 0.5;
const ALPHA_FIRST_RUN = 0.9;
// Was 0.07 — dwarfed by charge (-220) and link (0.2), so a source with few
// internal links (little to hold it together) scattered across the whole
// graph instead of reading as its own visual cluster; a source card's focus
// click also fits the camera to wherever its entities actually settled
// (fitToEntities), so a loosely-scattered source made that fit look almost
// like it hadn't moved. Raised enough to visibly pull a source together
// without overpowering real semantic links between well-connected entities
// (COG-6233).
const SOURCE_ANCHOR_STRENGTH = 0.16;
// Synchronous tick budget for presenting a NEW world (first load, dataset
// switch) already settled, instead of letting the user watch nodes fling
// into place for seconds. Growth within the SAME world (live updates)
// deliberately still animates — that motion is the product's "the graph
// grew" signal, not noise.
//
// d3's default alphaDecay (~0.0228) needs ~300 ticks to reach alphaMin,
// which benchmarks at ~380ms for 500 entities with this force config on an
// M-series in Node — over half a second of frozen main thread on a mid-range
// laptop. The steeper presettle decay converges in ~130 ticks instead, inside
// the budget below.
//
// There is no live simulation afterwards: the presettle drives the layout with
// synchronous sim.tick() calls (which, unlike the internal timer, dispatch no
// "tick"/"end" events) and then sim.stop() freezes it where it landed. That is
// the intended presentation — a new world appears settled instead of flinging
// into place — but it means the budget is the whole story: if changed forces or
// decay ever stop converging within PRESETTLE_MAX_TICKS, the layout freezes
// half-settled rather than continuing to relax on screen, so re-check
// convergence when touching any constant above. Note this runs in a useEffect,
// after paint — the seed scatter may be visible for a single frame before
// the block, which reads as a flicker at worst, not a scramble.
const PRESETTLE_ALPHA_DECAY = 0.05;
const PRESETTLE_MAX_TICKS = 150;

interface LinkDatum {
  source: string;
  target: string;
}

export function anchorOf(n: BusinessEntity, anchors: Record<string, Anchor>): Anchor {
  const sets = setsOf(n);
  if (!sets.length) return { x: 0, y: 0 };
  let x = 0, y = 0;
  sets.forEach((s) => {
    const a = anchors[s] || { x: 0, y: 0 };
    x += a.x;
    y += a.y;
  });
  return { x: x / sets.length, y: y / sets.length };
}

// Deterministic per-id scatter so a brand-new entity's starting position
// isn't always (0,0) before the simulation pulls it toward its anchor.
function hashId(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return h;
}

function seedPosition(
  n: BusinessEntity,
  anchors: Record<string, Anchor>,
  cached: [number, number] | undefined,
): void {
  if (cached) {
    [n.x, n.y] = cached;
    return;
  }
  const a = anchorOf(n, anchors);
  n.x = a.x + Math.sin(hashId(n.id)) * 150;
  n.y = a.y + Math.cos(hashId(n.id) * 2) * 110;
}

function idsKeyOf(entities: BusinessEntity[]): string {
  return entities.map((n) => n.id).sort().join(",");
}

// Seeds every entity's position (cached where seen before) and, when asked,
// collects the genuinely-new ones with their staggered birth timestamps.
function seedAndDetectNewborns(
  entities: BusinessEntity[],
  anchors: Record<string, Anchor>,
  importanceMax: number,
  positions: Record<string, [number, number]>,
  treatUncachedAsNewborn: boolean,
): { newborn: BusinessEntity[]; newbornAt: Record<string, number> } {
  const newborn: BusinessEntity[] = [];
  entities.forEach((n) => {
    n._r = 5 + 11 * ((n.importance || 0) / importanceMax);
    const cached = positions[n.id];
    seedPosition(n, anchors, cached);
    if (!cached && treatUncachedAsNewborn) newborn.push(n);
  });
  const born = performance.now();
  const newbornAt: Record<string, number> = {};
  newborn.forEach((n, i) => {
    newbornAt[n.id] = born + NEWBORN_BASE_DELAY_MS + i * NEWBORN_STAGGER_MS;
  });
  return { newborn, newbornAt };
}

// Ports startSimulation (customer_tutorial.html:6345). The reload-era
// localStorage position cache is gone — a query invalidation hands back a
// fresh `entities` array, and positionsRef (not localStorage) is what makes
// re-fetch look like a continuation: ids seen before keep their last
// position, ids that are genuinely new get staggered birth timing.
//
// useBrains refetches on an interval — every refetch hands back a brand-new
// `entities` array (freshly parsed JSON) even when the actual node set is
// unchanged. The simulation still needs to re-run on that fresh array (the
// new object instances need their x/y written back onto them), but reheating
// alpha as if it were a real change made every node visibly drift on every
// refetch — reported as the hover tooltip "fleeing" the node it was
// attached to, because the node itself was moving out from under the
// cursor. Comparing the id *set*, not the array reference, is what tells a
// same-data refetch (start cold, no visible motion) apart from an actual
// membership change (reheat and let it resettle).
export function useBusinessSimulation(
  entities: BusinessEntity[],
  semanticLinks: SemanticLink[],
  anchors: Record<string, Anchor>,
  importanceMax: number,
  onTick: () => void,
  onGrowth?: (newborn: BusinessEntity[]) => void,
  // Restricts which entities the physics forces actually run on (see
  // useViewportActiveIds) — null means "everyone", matching the previous,
  // uncapped behavior for every existing caller/dataset size. Entities left
  // out still get seeded/cached below, so they render at a valid position;
  // they just don't cost a force-simulation tick while off-screen.
  activeIds?: Set<string> | null,
  // Identity of the world being simulated (the focused dataset id). When it
  // changes — or on the very first run — the layout is computed synchronously
  // before the next paint (see PRESETTLE_MAX_TICKS) so a new dataset appears
  // already settled. Same-key updates (live growth, refetches) animate as
  // before.
  settleKey?: string | null,
): { newbornAt: Record<string, number> } {
  const positionsRef = useRef<Record<string, [number, number]>>({});
  const prevIdsKeyRef = useRef<string | null>(null);
  const prevSimulatedIdsKeyRef = useRef<string | null>(null);
  const prevSettleKeyRef = useRef<string | null | undefined>(undefined);
  const newbornAtRef = useRef<Record<string, number>>({});

  useEffect(() => {
    if (!entities.length) return;
    // Newborn/growth detection always looks at the FULL dataset, never the
    // viewport-capped subset below — whether an entity is genuinely new to
    // the graph has nothing to do with whether it's currently on-screen.
    const idsKey = idsKeyOf(entities);
    const isFirstRun = prevIdsKeyRef.current === null;
    const idsChanged = !isFirstRun && prevIdsKeyRef.current !== idsKey;
    // A different settleKey means a different WORLD (dataset switch), not the
    // same world growing — its entities are presented settled and all at
    // once (see PRESETTLE_MAX_TICKS below), never as staggered "newborns",
    // and it must not fire the growth narration. Both sides normalized to
    // null: the ref stores `settleKey ?? null`, so comparing a raw undefined
    // settleKey would read EVERY run as a new world — a synchronous
    // presettle on each refetch and viewport pan.
    const settleKeyNow = settleKey ?? null;
    const isNewWorld = isFirstRun || prevSettleKeyRef.current !== settleKeyNow;

    const { newborn, newbornAt } = seedAndDetectNewborns(
      entities, anchors, importanceMax, positionsRef.current, idsChanged && !isNewWorld,
    );
    newbornAtRef.current = newbornAt;
    // Ports the reload-era "cognify complete" narration's gate (customer_
    // tutorial.html ~7215: "if (newborn.length) narrate(...)") — this port's
    // refetch already re-seeds without a reload, but the caller still wants
    // to know a real membership change (not just a same-data refetch) just
    // happened, to narrate it and flash the source it came from.
    if (newborn.length) onGrowth?.(newborn);

    // Physics only runs on the active subset (everyone, when activeIds is
    // null) — reheat is judged against ITS membership too, separately from
    // idsChanged above, so panning into a new area (which changes who's
    // active without any real dataset growth) still gets a mild reheat
    // instead of settling at whatever alpha the last, unrelated dataset
    // change left behind.
    const simulatedEntities = activeIds ? entities.filter((n) => activeIds.has(n.id)) : entities;
    const simulatedIdsKey = idsKeyOf(simulatedEntities);
    const simulatedSetChanged = prevSimulatedIdsKeyRef.current !== null && prevSimulatedIdsKeyRef.current !== simulatedIdsKey;
    const willReheat = isFirstRun || idsChanged || simulatedSetChanged;

    const linkDatums: LinkDatum[] = semanticLinks
      .filter((l) => !activeIds || (activeIds.has(l._sid) && activeIds.has(l._tid)))
      .map((l) => ({ source: l._sid, target: l._tid }));
    const sim = forceSimulation<BusinessEntity>(simulatedEntities)
      .force(
        "link",
        forceLink<BusinessEntity, LinkDatum>(linkDatums)
          .id((d) => d.id)
          .distance(110)
          .strength(0.2),
      )
      .force("charge", forceManyBody<BusinessEntity>().strength(-220))
      .force(
        "collide",
        forceCollide<BusinessEntity>().radius((d) => (d._r ?? 5) + 26),
      )
      .force("ax", forceX<BusinessEntity>((d) => anchorOf(d, anchors).x).strength(SOURCE_ANCHOR_STRENGTH))
      .force("ay", forceY<BusinessEntity>((d) => anchorOf(d, anchors).y).strength(SOURCE_ANCHOR_STRENGTH))
      .alpha(isFirstRun ? ALPHA_FIRST_RUN : willReheat ? ALPHA_CHANGED_IDS : ALPHA_UNCHANGED_IDS)
      .on("tick", onTick);

    if (isNewWorld) {
      // Stop first so the internal timer never animates this world, tick it to
      // convergence by hand, then leave it stopped: nothing restarts it, so the
      // frozen layout IS the result (see PRESETTLE_ALPHA_DECAY). Restoring the
      // live decay afterwards only keeps the instance truthfully configured for
      // whoever reads or restarts it.
      sim.stop();
      const liveDecay = sim.alphaDecay();
      sim.alphaDecay(PRESETTLE_ALPHA_DECAY);
      let presettleTicks = 0;
      while (sim.alpha() > sim.alphaMin() && presettleTicks < PRESETTLE_MAX_TICKS) {
        sim.tick();
        presettleTicks += 1;
      }
      sim.alphaDecay(liveDecay);
    }

    const flushPositions = (): void => {
      simulatedEntities.forEach((n) => {
        if (typeof n.x === "number" && typeof n.y === "number") {
          positionsRef.current[n.id] = [n.x, n.y];
        }
      });
    };
    sim.on("end", flushPositions);

    prevIdsKeyRef.current = idsKey;
    prevSimulatedIdsKeyRef.current = simulatedIdsKey;
    prevSettleKeyRef.current = settleKeyNow;

    return () => {
      // Positions were only ever cached on "end" — a restart before the sim
      // settles (a same-data refetch landing mid-alpha, or panning
      // activeIds every ~800ms on a large graph) discarded in-progress
      // motion, so the next run's seedPosition fell back to the deterministic
      // hash-scatter and re-exploded the whole graph instead of continuing
      // from where it visually was (COG-6233). A stop is now as good as an
      // end for caching purposes.
      flushPositions();
      sim.stop();
    };
  }, [entities, semanticLinks, anchors, importanceMax, onTick, onGrowth, activeIds, settleKey]);

  return { newbornAt: newbornAtRef.current };
}

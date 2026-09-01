"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, type MouseEvent, type RefObject } from "react";
import type { BrainState, BusinessEntity } from "../sceneTypes";
import { useBusinessSimulation } from "../simulation/useBusinessSimulation";
import { useViewportActiveIds } from "../simulation/useViewportActiveIds";
import type { ZoomTransform } from "d3";
import { useBusinessCamera, computeFitScale } from "./useBusinessCamera";
import { useCanvasSize } from "./useCanvasSize";
import { hitTestScene, hitTestAgentMarker, type SceneHit } from "./businessHitTest";
import { draw, computeTypeFadeKMax, computeInstanceAlpha, type Spotlight } from "./businessDraw";
import { drawFilaments, filamentTargets, sourcesWithEntities } from "./businessFilaments";
import {
  computeAgentPresence, drawAgentPresence, markerHitRadiusWorld, type AgentPresence, type AskingAgent,
} from "./businessAgentPresence";
import { drawAmbientBackground } from "./businessAmbientBackground";
import { drawMinimap, minimapWorldPoint, minimapContains } from "./businessMinimap";
import { usePrefersReducedMotion } from "../usePrefersReducedMotion";
import {
  isSpotlightActive, cardScreenPosition, EMPTY_BRAIN_STATE, EMPTY_ANSWERED_IDS,
  EMPTY_PATH_IDS, EMPTY_PATH_EDGE_KEYS,
} from "./businessCanvasHelpers";

type CardRefMap = RefObject<Record<string, HTMLElement | null>>;

export interface BusinessCanvasHandle {
  fit: (animate: boolean) => void;
  goToAltimeterLevel: (level: number) => void;
  // Ports playSearchEvent's "camera to the retrieved subgraph's bbox"
  // (customer_tutorial.html ~7060) — same fit-to-content math as `fit`, just
  // scoped to the ids a search hit rather than every entity.
  focusOnIds: (ids: Set<string>) => void;
  // Ports _bvLiveEvent's idle check ("never steal the camera mid-
  // interaction... dock a quiet chip instead") — Infinity when the user has
  // never touched the camera at all, matching lastInteraction===0's
  // "always auto-focus" sentinel in the source.
  getIdleMs: () => number;
  // Snapshot/restore for interruptions (the tour): capture the transform at
  // start, put it back instantly — no animation — when the user interrupts.
  getTransform: () => ZoomTransform;
  setTransformNow: (t: ZoomTransform) => void;
}

interface BusinessCanvasProps {
  brainState: BrainState | null;
  selectedId: string | null;
  // shiftKey lets BusinessView tell a plain click (select this, replacing
  // whatever was selected) apart from a shift-click (trace the shortest
  // path from the CURRENT selection to this one) — see useShortestPath.
  onSelectEntity: (entity: BusinessEntity, shiftKey: boolean) => void;
  onHover: (hit: SceneHit) => void;
  // Fired on every mousemove with canvas-local coordinates (and the
  // canvas's own width, for edge-clamping) — the canvas is the only element
  // in this tree that reliably receives pointer events here, so it's also
  // the only reliable source of "where is the cursor" for the tooltip.
  onHoverMove?: (x: number, y: number, containerWidth: number) => void;
  onBackgroundClick: () => void;
  spotlight: Spotlight | null;
  focusSets: Set<string> | null;
  onLevelChange?: (level: number, plumbing: boolean) => void;
  sourceCardRefs?: CardRefMap;
  activeDatasetId?: string | null;
  // Ports the reload-era "cognify complete" narration's trigger
  // (customer_tutorial.html ~7215) — fires when the simulation sees ids it
  // hasn't seen before, distinct from a same-data refetch.
  onGrowth?: (newborn: BusinessEntity[]) => void;
  // Entities that have EVER answered a live search, across the whole
  // session's event log — a lasting fact, not the current 9s spotlight.
  answeredIds?: Set<string>;
  // The shortest path currently being traced (useShortestPath) — empty
  // when no path is active.
  pathIds?: Set<string>;
  pathEdgeKeys?: Set<string>;
  // CLO-606: the agent a live search answer is currently attributed to (with
  // the window that attribution is good for), or null when no agent is
  // asking. Drawing the marker additionally requires that the spotlight on
  // screen is one the agent produced — computeAgentPresence checks the
  // spotlight's own `source`, since the auto-insight and what-if-removal
  // spotlights set `spotlight` with no agent involved.
  askingAgent?: AskingAgent | null;
  // The principal OperatorsRail (or this canvas's own marker) currently has
  // hovered — drawn back into the marker as an emphasized ring when it
  // matches the asking agent's id (rail→canvas half of the hover sync).
  hoveredPrincipalId?: string | null;
  // Canvas→rail half of the same sync: fires with the asking agent's id
  // while the marker itself is hovered, null when it isn't — the caller
  // feeds this into the same state OperatorsRail's own hover already sets.
  onHoverAgent?: (agentId: string | null) => void;
}

// Ports the canvas half of the Business view: the draw loop draws through
// refs (transform, level, hovered/newborn timing) so React state changes
// elsewhere in the page never remount this component and reset the
// animation — the constraint the ticket calls the single biggest risk in
// the port. useBusinessCamera returns a fresh object every render (it's a
// hook, not a ref itself), so this destructures its ref/callback members up
// front and depends on THOSE — each one is stable across renders — rather
// than the whole `camera` object, which would tear down and restart the RAF
// loop below on every unrelated re-render of this component.
const BusinessCanvas = forwardRef<BusinessCanvasHandle, BusinessCanvasProps>(function BusinessCanvas(
  {
    brainState, selectedId, onSelectEntity, onHover, onHoverMove, onBackgroundClick, spotlight, focusSets,
    onLevelChange, sourceCardRefs, activeDatasetId, onGrowth,
    answeredIds, pathIds, pathEdgeKeys,
    askingAgent, hoveredPrincipalId, onHoverAgent,
  },
  ref,
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hoveredIdRef = useRef<string | null>(null);
  const lastHoverKeyRef = useRef<string | null>(null);
  const rafRef = useRef<number | null>(null);
  // The cursor's last known canvas-local position, re-hit-tested every RAF
  // frame below — not just on the native mousemove that set it. Entities are
  // live simulation nodes that keep moving for a second or two after any
  // fetch that changes the id set (a fresh dataset, in particular, always
  // does — see useBusinessSimulation's ALPHA_CHANGED_IDS), so a hover
  // resolved only on mousemove goes stale mid-settle: the ring keeps
  // following the entity as it drifts across the screen while the tooltip
  // (anchored to the cursor, which hasn't moved) stays put, reading as the
  // highlight "fleeing" the mouse. Re-running the same hit test on every
  // frame keeps the hover locked to whatever entity is actually under the
  // cursor right now, mouse-still or not.
  const lastMouseRef = useRef<{ x: number; y: number } | null>(null);
  const onHoverRef = useRef(onHover);
  onHoverRef.current = onHover;
  // Same "draw through refs" reasoning as onHoverRef above: these change on
  // ordinary user actions (a spotlight tick, a live-search poll producing a
  // new answeredIds Set) that have nothing to do with the canvas element
  // itself — closing over them directly in the RAF effect's deps below tore
  // the persistent loop down and rebuilt it on each one, contrary to this
  // file's own stated design (COG-6233).
  const spotlightRef = useRef(spotlight);
  spotlightRef.current = spotlight;
  const focusSetsRef = useRef(focusSets);
  focusSetsRef.current = focusSets;
  const selectedIdRef = useRef(selectedId);
  selectedIdRef.current = selectedId;
  const answeredIdsRef = useRef(answeredIds);
  answeredIdsRef.current = answeredIds;
  const pathIdsRef = useRef(pathIds);
  pathIdsRef.current = pathIds;
  const pathEdgeKeysRef = useRef(pathEdgeKeys);
  pathEdgeKeysRef.current = pathEdgeKeys;
  // Same "draw through refs" reasoning as spotlightRef above — CLO-606's
  // agent presence.
  const askingAgentRef = useRef(askingAgent);
  askingAgentRef.current = askingAgent;
  const hoveredPrincipalIdRef = useRef(hoveredPrincipalId);
  hoveredPrincipalIdRef.current = hoveredPrincipalId;
  const onHoverAgentRef = useRef(onHoverAgent);
  onHoverAgentRef.current = onHoverAgent;
  // The last id reported through onHoverAgent, so it only fires on a real
  // change — the same pattern lastHoverKeyRef uses for entity/type hover.
  const lastAgentHoverIdRef = useRef<string | null>(null);
  // Last frame's marker geometry — the marker is drawn from a plain local in
  // the RAF tick (hit-test and draw agree within a frame that way), but the
  // click handler runs outside the loop and still has to know whether the
  // click landed on the marker: without that it fell through to
  // onBackgroundClick, which clears the spotlight, so clicking the marker
  // deleted the marker.
  const agentMarkerRef = useRef<{ presence: AgentPresence; hitRadiusWorld: number } | null>(null);
  // Ids drawEntities actually drew last frame (see businessDraw.draw) — hit
  // testing filters to this set first, so a record hidden by the focus lens,
  // the below-L1 importance cut, or a not-yet-faded-in newborn never shows a
  // tooltip or accepts a click just because its world position is under the
  // cursor.
  const visibleEntityIdsRef = useRef<Set<string>>(new Set());
  // Last frame's schema-crossfade threshold (fit-scale-relative, see the RAF
  // tick) — the click handler reads it so click-time type hit-testing agrees
  // with what that frame actually drew.
  const typeFadeKMaxRef = useRef(1.55);

  const { width, height, dpr } = useCanvasSize(containerRef);
  const reducedMotion = usePrefersReducedMotion();
  // The simulation's onTick has nothing to schedule — a persistent RAF loop
  // below already redraws every frame, reading node positions the
  // simulation mutates in place.
  const noopTick = useCallback(() => {}, []);

  const brain = brainState ?? EMPTY_BRAIN_STATE;
  // Recomputed only when entities change, then read through a ref by the
  // persistent rAF draw loop below — computing it per frame allocated a Set
  // and walked every entity 60 times a second for a per-dataset constant.
  const namesWithEntities = useMemo(() => sourcesWithEntities(brain.entities), [brain.entities]);
  const namesWithEntitiesRef = useRef(namesWithEntities);
  namesWithEntitiesRef.current = namesWithEntities;
  const { transformRef, levelRef, plumbingRef, lastInteractionRef, fitToEntities, goToAltimeterLevel, centerOnWorld, applyTransform } =
    useBusinessCamera(canvasRef, onLevelChange);
  const activeIds = useViewportActiveIds(brain.entities, transformRef, width, height);
  const { newbornAt } = useBusinessSimulation(
    brain.entities,
    brain.semanticLinks,
    brain.anchors,
    brain.importanceMax,
    noopTick,
    onGrowth,
    activeIds,
    activeDatasetId,
  );

  useImperativeHandle(ref, () => ({
    fit: (animate) => fitToEntities(brain.entities, width, height, animate),
    goToAltimeterLevel: (level) => goToAltimeterLevel(level, width, height, brain.entities, focusSetsRef.current),
    focusOnIds: (ids) => fitToEntities(brain.entities.filter((n) => ids.has(n.id)), width, height, true),
    getIdleMs: () => (lastInteractionRef.current === 0 ? Infinity : performance.now() - lastInteractionRef.current),
    getTransform: () => transformRef.current,
    setTransformNow: (t) => applyTransform(t, false),
  }));

  // Auto-fits whenever the focused dataset actually changes, instead of the
  // caller firing fit() off a requestAnimationFrame right after switching
  // datasets. That external rAF raced useBusinessSimulation's seeding effect
  // above: on a genuinely new dataset, brand-new entities start with no x/y
  // at all (seedPosition hasn't run yet), so a fit that reads them first
  // measures a zero-size bbox at the origin and locks the camera there —
  // then seeding moves the entities out to their real anchor positions,
  // which land wherever that anchor happens to be (often off in a corner)
  // since the camera never re-measures. Declaring this effect after the
  // useBusinessSimulation() call above guarantees React flushes it after
  // that hook's own effect in the same commit, so entities are always
  // already seeded by the time fitToEntities measures them here.
  const lastFitRef = useRef<{ datasetId: string | null; hadEntities: boolean } | undefined>(undefined);
  useEffect(() => {
    // Fitting with zero entities is still deliberate (a dataset with zero
    // entity-stage nodes never grows any, and skipping the fit left the
    // camera on a PREVIOUS dataset's transform — COG-6233; fitToEntities
    // recenters on world origin when empty). But since the per-dataset
    // fetch (CLO-597), a cold switch renders governance-only FIRST and the
    // graph arrives a moment later — so an empty fit must not count as
    // final: track whether the fit saw entities, and fit once more when the
    // same dataset transitions from empty to populated, or the camera never
    // frames the graph that just landed.
    if (width === 0 || height === 0) return;
    const datasetId = activeDatasetId ?? null;
    const hasEntities = brain.entities.length > 0;
    const prev = lastFitRef.current;
    if (prev && prev.datasetId === datasetId && (prev.hadEntities || !hasEntities)) return;
    const isFirstFit = prev === undefined;
    lastFitRef.current = { datasetId, hadEntities: hasEntities };
    fitToEntities(brain.entities, width, height, !isFirstFit);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeDatasetId, brain.entities, width, height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width === 0 || height === 0) return;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }, [width, height, dpr]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const tick = (): void => {
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const now = performance.now();
      drawAmbientBackground(ctx, width, height, now, reducedMotion);
      const canvasRect = canvas.getBoundingClientRect();
      // Entities move every simulation tick, so the fit scale (and with it
      // the schema crossfade threshold) is a per-frame quantity — cheap, the
      // minimap already walks the same array each frame anyway.
      const typeFadeKMax = computeTypeFadeKMax(computeFitScale(brain.entities, width, height));
      typeFadeKMaxRef.current = typeFadeKMax;

      // CLO-606: computed once here, ahead of the mouse hit-test block below
      // (so hit-testing and the actual draw call further down agree within
      // this one frame) — null whenever no agent is asking or the spotlight
      // that would gate it has expired, which is also this feature's "no
      // permanent lines" rule: nothing draws once this is null.
      const spotlightNow = spotlightRef.current;
      const askingAgentNow = askingAgentRef.current;
      const agentPresence = askingAgentNow && spotlightNow
        ? computeAgentPresence(brain.entities, spotlightNow, askingAgentNow, now, transformRef.current.k)
        : null;
      const agentEmphasized = Boolean(askingAgentNow && hoveredPrincipalIdRef.current === askingAgentNow.id);
      // Same radius drawMarkerDot is about to use, so the target and the dot
      // agree exactly — including the pulse, which a flat threshold could
      // only ever match at one point in its cycle.
      agentMarkerRef.current = agentPresence
        ? {
            presence: agentPresence,
            hitRadiusWorld: markerHitRadiusWorld(transformRef.current.k, reducedMotion, now, agentEmphasized),
          }
        : null;

      if (lastMouseRef.current) {
        // The minimap floats over the scene — entities underneath it should
        // never light up or tooltip while the cursor is on the panel.
        const overMinimap = minimapContains(lastMouseRef.current.x, lastMouseRef.current.y, width, height);
        const hit = overMinimap ? null : hitTestScene(
          lastMouseRef.current.x,
          lastMouseRef.current.y,
          transformRef.current,
          brain.typeNodes,
          brain.entities.filter((n) => visibleEntityIdsRef.current.has(n.id)),
          isSpotlightActive(spotlightRef.current),
          hoveredIdRef.current,
          typeFadeKMax,
        );
        hoveredIdRef.current = hit?.kind === "entity" ? hit.node.id : null;
        const key = hit ? `${hit.kind}:${hit.kind === "entity" ? hit.node.id : hit.node.name}` : null;
        if (key !== lastHoverKeyRef.current) {
          lastHoverKeyRef.current = key;
          onHoverRef.current(hit);
        }

        // The marker is bigger-picture context — an entity or type node
        // under the same cursor wins, and the minimap guard above applies
        // here too.
        const marker = agentMarkerRef.current;
        const agentHit = !overMinimap && !hit && marker
          ? hitTestAgentMarker(
              marker.presence.markerWorld,
              lastMouseRef.current.x,
              lastMouseRef.current.y,
              transformRef.current,
              marker.hitRadiusWorld,
            )
          : false;
        const agentHoverId = agentHit && askingAgentNow ? askingAgentNow.id : null;
        if (agentHoverId !== lastAgentHoverIdRef.current) {
          lastAgentHoverIdRef.current = agentHoverId;
          onHoverAgentRef.current?.(agentHoverId);
        }
      }

      const cardMap = sourceCardRefs?.current;
      // A source with zero extracted entities and zero links has nothing to
      // thread a filament to — buildAnchors puts a lone source's anchor at
      // world origin (0,0), and with an empty entity set fitToEntities
      // recenters the camera there too, so the filament used to draw a
      // stray line from the source card straight to screen-center.
      const hasGraphContent = brain.entities.length > 0 || brain.semanticLinks.length > 0;
      // Filaments thread to entity-cluster centroids, so they only mean
      // anything while the entity layer is actually on screen — past the
      // schema crossfade those clusters aren't drawn at all and the threads
      // read as lines crossing the canvas toward nothing. Same treatment
      // drawSourceHulls gets inside draw() for the same reason.
      const filamentAlpha = computeInstanceAlpha(
        transformRef.current.k,
        typeFadeKMax,
        brain.typeNodes.length > 0,
        isSpotlightActive(spotlightRef.current),
      );
      if (cardMap && brain.sourceNames.length && hasGraphContent && filamentAlpha > 0.01) {
        const cardPositions: Record<string, { x: number; y: number }> = {};
        // Filaments follow the focus lens: a filtered-out source's entities
        // are hidden, so its thread to an empty territory read as stray
        // lines crossing the canvas from nowhere to nowhere.
        const focusSetsNow = focusSetsRef.current;
        // sourceNames also includes dataset sources with zero extracted
        // entities (COG-6233) — skip those or the filament threads to a
        // fallback anchor with nothing real on the other end.
        const namesWithEntitiesNow = namesWithEntitiesRef.current;
        const filamentNames = (focusSetsNow
          ? brain.sourceNames.filter((name) => focusSetsNow.has(name))
          : brain.sourceNames
        ).filter((name) => namesWithEntitiesNow.has(name));
        filamentNames.forEach((name) => {
          const pos = cardScreenPosition(cardMap, name, canvasRect);
          if (pos) cardPositions[name] = pos;
        });
        drawFilaments(ctx, cardPositions, filamentTargets(brain.entities, brain.anchors), transformRef.current, now, filamentAlpha);
      }

      ctx.translate(transformRef.current.x, transformRef.current.y);
      ctx.scale(transformRef.current.k, transformRef.current.k);
      draw(ctx, {
        transform: transformRef.current,
        level: plumbingRef.current ? 3 : levelRef.current,
        entities: brain.entities,
        typeNodes: brain.typeNodes,
        typeLinks: brain.typeLinks,
        semanticLinks: brain.semanticLinks,
        docLinks: brain.docLinks,
        byId: brain.byId,
        setColor: brain.setColor,
        sourceNames: brain.sourceNames,
        importanceMax: brain.importanceMax,
        isSessionSet: brain.isSessionSet,
        newbornAt,
        hoveredId: hoveredIdRef.current,
        selectedId: selectedIdRef.current,
        spotlight: spotlightRef.current,
        focusSets: focusSetsRef.current,
        reducedMotion,
        answeredIds: answeredIdsRef.current ?? EMPTY_ANSWERED_IDS,
        pathIds: pathIdsRef.current ?? EMPTY_PATH_IDS,
        pathEdgeKeys: pathEdgeKeysRef.current ?? EMPTY_PATH_EDGE_KEYS,
        plumbingNodes: brain.plumbingNodes,
        plumbingEntityId: brain.plumbingEntityId,
        typeFadeKMax,
        importanceCut: brain.importanceCut,
        connectedIds: brain.connectedIds,
      }, now, visibleEntityIdsRef.current);
      // Drawn inside the same camera transform as the entity/type layers
      // above, so the marker/filament/label pan and zoom with the cluster
      // they point at — and after them, so the marker never sits under a
      // node's own drawing.
      if (agentPresence && askingAgentNow) {
        drawAgentPresence(ctx, {
          presence: agentPresence,
          name: askingAgentNow.name,
          transformK: transformRef.current.k,
          reducedMotion,
          emphasized: agentEmphasized,
          now,
        });
      }
      ctx.restore();
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // draw() above just repopulated visibleEntityIdsRef for this exact
      // frame — filtering to it here keeps the minimap's dots and its own
      // world-bounds projection scoped to whatever's currently shown, so
      // panel filters (a source, a focus lens) aren't contradicted by a
      // minimap still tracking the whole unfiltered graph.
      const minimapEntities = brain.entities.filter((n) => visibleEntityIdsRef.current.has(n.id));
      drawMinimap(ctx, minimapEntities, brain.setColor, transformRef.current, width, height);
      ctx.restore();
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [
    // selectedId/spotlight/focusSets/answeredIds/pathIds/pathEdgeKeys/
    // activeDatasetId are deliberately NOT here — they're read through the
    // refs synced above, so this persistent RAF loop survives their changes
    // instead of being torn down and rebuilt on every one (COG-6233; see
    // this component's own header comment on why refs exist).
    brain, transformRef, levelRef, plumbingRef, dpr, width, height, newbornAt,
    reducedMotion, sourceCardRefs,
  ]);

  const handleMouseMove = useCallback(
    (ev: MouseEvent<HTMLCanvasElement>) => {
      // The actual hit test runs in the RAF tick above (every frame, off
      // this same position) so a still cursor stays in sync with a moving
      // entity — this handler only records where the cursor is and reports
      // the tooltip's screen position, which is cursor-anchored and needs
      // no re-test of its own.
      lastMouseRef.current = { x: ev.nativeEvent.offsetX, y: ev.nativeEvent.offsetY };
      onHoverMove?.(ev.nativeEvent.offsetX, ev.nativeEvent.offsetY, width);
    },
    [onHoverMove, width],
  );

  const handleClick = useCallback(
    (ev: MouseEvent<HTMLCanvasElement>) => {
      // Same filtered set drawMinimap projects from, so a click lands where
      // the panel's dots visually are, not where the unfiltered graph would
      // have projected them.
      const minimapEntities = brain.entities.filter((n) => visibleEntityIdsRef.current.has(n.id));
      const target = minimapWorldPoint(ev.nativeEvent.offsetX, ev.nativeEvent.offsetY, minimapEntities, width, height);
      if (target) {
        centerOnWorld(target.x, target.y, width, height);
        return;
      }
      const hit = hitTestScene(
        ev.nativeEvent.offsetX,
        ev.nativeEvent.offsetY,
        transformRef.current,
        brain.typeNodes,
        brain.entities.filter((n) => visibleEntityIdsRef.current.has(n.id)),
        isSpotlightActive(spotlight),
        undefined,
        typeFadeKMaxRef.current,
      );
      if (hit?.kind === "entity") {
        onSelectEntity(hit.node, ev.shiftKey);
        return;
      }
      // The marker is not a control — it annotates, it has no click action.
      // Falling through to onBackgroundClick made it worse than inert: that
      // clears the spotlight the marker is gated on, so clicking the thing
      // the emphasis ring invites you to click made it vanish.
      const marker = agentMarkerRef.current;
      if (marker && hitTestAgentMarker(
        marker.presence.markerWorld,
        ev.nativeEvent.offsetX,
        ev.nativeEvent.offsetY,
        transformRef.current,
        marker.hitRadiusWorld,
      )) return;
      onBackgroundClick();
    },
    [transformRef, brain, spotlight, onSelectEntity, onBackgroundClick, centerOnWorld, width, height],
  );

  return (
    // overscroll-none: without it, a trackpad zoom/pan gesture's leftover
    // horizontal delta (d3-zoom consumes the wheel event for zooming, but
    // that doesn't stop the browser's own edge-overscroll tracking) bubbles
    // to the document — after a few zoom in/out cycles it accumulates
    // enough to trigger the browser's native swipe-back navigation, kicking
    // the user out of Business entirely to whatever page they visited before.
    <div ref={containerRef} className="bv-canvas-layer absolute inset-0 overscroll-none">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 cursor-grab overscroll-none"
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        onMouseLeave={() => {
          lastMouseRef.current = null;
          hoveredIdRef.current = null;
          lastHoverKeyRef.current = null;
          onHover(null);
          if (lastAgentHoverIdRef.current !== null) {
            lastAgentHoverIdRef.current = null;
            onHoverAgent?.(null);
          }
        }}
      />
    </div>
  );
});

export default BusinessCanvas;

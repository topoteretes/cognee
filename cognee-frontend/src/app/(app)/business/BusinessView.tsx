"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { notifications } from "@mantine/notifications";
import type { CogneeInstance } from "@/modules/instances/types";
import { useBusinessScene } from "@/modules/business/useBusinessScene";
import { useBrains } from "@/modules/business/useBrains";
import { useGovernanceIndex } from "@/modules/business/useGovernanceIndex";
import { useBusinessLiveUpdates } from "@/modules/business/useBusinessLiveUpdates";
import { computeViewingNarration } from "@/modules/business/viewingNarration";
import BusinessCanvas, { type BusinessCanvasHandle } from "@/modules/business/canvas/BusinessCanvas";
import type { Spotlight } from "@/modules/business/canvas/businessDraw";
import type { BusinessEntity } from "@/modules/business/sceneTypes";
import HoverTooltip from "@/modules/business/panels/HoverTooltip";
import type { SceneHit } from "@/modules/business/canvas/businessHitTest";
import NodePanel from "@/modules/business/panels/NodePanel";
import SourceDetailCard from "@/modules/business/panels/SourceDetailCard";
import { computeSourceDetail } from "@/modules/business/computeSourceDetail";
import { computeWhatIfRemoval } from "@/modules/business/computeWhatIfRemoval";
import OperatorsRail from "@/modules/business/panels/OperatorsRail";
import BrainSwitcher from "@/modules/business/panels/BrainSwitcher";
import SourcesRail from "@/modules/business/panels/SourcesRail";
import GraphConstructionLog, { type ConstructionLogEntry } from "@/modules/business/panels/GraphConstructionLog";
import AnswerCard from "@/modules/business/panels/AnswerCard";
import SessionMemoryCard from "@/modules/business/panels/SessionMemoryCard";
import { GOVERNANCE_LAYER_ID } from "@/modules/business/layers/governanceLayer";
import { CONTENT_LAYER_ID } from "@/modules/business/layers/contentLayer";
import { accessibleDatasetIds } from "@/modules/business/useGovernanceIndex";
import { setsOf, sourceLabel } from "@/modules/business/computeBrainState";
import { truncate } from "@/modules/business/textUtils";
import { useBusinessQaSurface } from "@/modules/business/useBusinessQaSurface";
import { useNarration } from "@/modules/business/useNarration";
import { useBusinessTour } from "@/modules/business/useBusinessTour";
import { useBusinessAutoInsights } from "@/modules/business/useBusinessAutoInsights";
import SearchBar from "@/modules/business/panels/SearchBar";
import BusinessLegend from "@/modules/business/panels/BusinessLegend";
import BusinessDock from "@/modules/business/panels/BusinessDock";
import BusinessLoading from "@/modules/business/panels/BusinessLoading";
import BusinessEmptyState from "@/modules/business/panels/BusinessEmptyState";
import ScrollFadeContainer from "@/modules/business/panels/ScrollFadeContainer";
import { useEntitySelection } from "@/modules/business/useEntitySelection";

const WHAT_IF_SPOTLIGHT_MS = 12000;

interface BusinessViewProps {
  cogniInstance: CogneeInstance;
}

// Requires a ready cogniInstance so every hook below can assume it's real —
// CustomAppShell already shows the pod-provisioning state for "/business"
// before this ever mounts (see POD_DEPENDENT_PATHS), the same gate every
// other pod-backed route uses.
export default function BusinessView({ cogniInstance }: BusinessViewProps) {
  const scene = useBusinessScene(cogniInstance);
  const brainsQuery = useBrains(cogniInstance);
  const governanceIndex = useGovernanceIndex(scene.layerData[GOVERNANCE_LAYER_ID]);
  const { display: narration, narrate } = useNarration();
  const live = useBusinessLiveUpdates(scene.activeDatasetId, cogniInstance);

  const canvasRef = useRef<BusinessCanvasHandle>(null);
  const selection = useEntitySelection(scene.brainState?.semanticLinks, scene.brainState?.entityById);
  const [hoveredHit, setHoveredHit] = useState<SceneHit>(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0, containerWidth: 0 });
  const [hoveredPrincipalId, setHoveredPrincipalId] = useState<string | null>(null);
  // Kept apart from hoveredPrincipalId on purpose. The rail's hover is an
  // access question ("what can this principal read"), so it dims the sources
  // rail; hovering the canvas marker is just "which agent is that", and
  // feeding it into the same state greyed the entire sources rail out from
  // under the user for pointing at a dot.
  const [hoveredMarkerAgentId, setHoveredMarkerAgentId] = useState<string | null>(null);
  const [focusSets, setFocusSets] = useState<Set<string> | null>(null);
  const [spotlight, setSpotlight] = useState<Spotlight | null>(null);
  // UI-facing mirror of the canvas's ref-based level/plumbing (see
  // useBusinessCamera's onLevelChange) — only for highlighting the active
  // altimeter button, never read by the draw loop itself.
  const [altimeterState, setAltimeterState] = useState({ level: 0, plumbing: false });
  const [flashedSourceName, setFlashedSourceName] = useState<string | null>(null);
  // Tracks which source the info card is showing — separate from focusSets
  // because that state is shared with the session-memory lens (both call
  // setFocusSets), so it can't tell "focused on one source" apart from
  // "focused on one distilled session set" on its own.
  const [selectedSourceName, setSelectedSourceName] = useState<string | null>(null);
  const [constructionLog, setConstructionLog] = useState<ConstructionLogEntry[]>([]);
  const constructionLogSeq = useRef(0);
  const sourceCardRefs = useRef<Record<string, HTMLElement | null>>({});
  const registerSourceCardRef = useCallback((name: string, el: HTMLElement | null) => {
    sourceCardRefs.current[name] = el;
  }, []);
  // Auto-focus the first readable brain once the list loads, so the canvas
  // never sits empty on first paint waiting for a manual pick.
  //
  // Driven by the governance list, not by /visualize/brains: that endpoint
  // returns every readable dataset's FULL graph in one response, so on a
  // workspace with a dozen populated datasets it routinely exceeds the
  // client's 10s GET budget and rejects. Hanging auto-select off it meant a
  // slow-but-healthy workspace landed on "select a brain" with an empty
  // canvas and no loader, while the governance panel beside it rendered
  // fine (reported live on an 11-dataset workspace). Governance comes from
  // its own light endpoint, already carries every dataset, and exposes them
  // with bare ids precisely so they can be used as activeDatasetId (see
  // useGovernanceIndex's bareDatasetId).
  const firstGovernanceDatasetId = governanceIndex.datasets[0]?.id ?? null;
  useEffect(() => {
    if (scene.activeDatasetId || !firstGovernanceDatasetId) return;
    scene.setActiveDatasetId(firstGovernanceDatasetId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [firstGovernanceDatasetId, scene.activeDatasetId]);

  // Ports the tab-activation gate's one-time narration (customer_tutorial
  // .html ~6188). Fitting the camera to the first dataset is BusinessCanvas's
  // own job now (see its activeDatasetId effect) — it can order itself
  // after the simulation's seeding effect in a way a rAF fired from here
  // never could, since React only guarantees hook-effect order within the
  // same component. This fires exactly once, whenever brainState first
  // appears, so the narration doesn't repeat on later dataset switches.
  const hasNarratedFirstLoadRef = useRef(false);
  useEffect(() => {
    // Wait for the focused dataset's own graph, not for /visualize/brains:
    // brainState turns non-null as soon as the governance layer resolves, and
    // narrating then locked this one-time line onto "0 kinds of things across
    // 0 sources" forever even after real data arrived (COG-6233). The content
    // layer's payload is the honest signal for "the graph is here" — and
    // unlike the brains fetch it can't stall this behind a heavy
    // every-dataset response that may never arrive at all.
    if (hasNarratedFirstLoadRef.current || !scene.brainState || !scene.layerData[CONTENT_LAYER_ID]) return;
    hasNarratedFirstLoadRef.current = true;
    const brain = scene.brainState;
    narrate(
      `this is your business — ${brain.typeNodes.length} kind${brain.typeNodes.length === 1 ? "" : "s"} of things across ${brain.sourceNames.length} source${brain.sourceNames.length === 1 ? "" : "s"}, one connected model`,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene.brainState, scene.layerData]);

  // Ports the reload-era "cognify complete" narration + source-card flash
  // (customer_tutorial.html ~7215-7220) — this port's refetch re-seeds the
  // simulation on every poll regardless, so useBusinessSimulation only
  // calls this when the id SET actually grew, not on every same-data tick.
  const handleGrowth = useCallback((newborn: BusinessEntity[]) => {
    const bySet: Record<string, number> = {};
    newborn.forEach((n) => {
      const first = setsOf(n)[0];
      if (first) bySet[first] = (bySet[first] || 0) + 1;
    });
    const src = Object.keys(bySet).sort((a, b) => bySet[b] - bySet[a])[0];
    const text = `cognify complete — ${newborn.length} new entities joined the model${src ? ` from ${src}` : ""}`;
    narrate(text, "#43D9E8");
    // A real state change gets its own dismissible toast, not just the
    // narration line — that line is shared with passive auto-insight tips
    // (see useBusinessAutoInsights) and fades on its own, so a genuine event
    // was easy to mistake for one of those tips or miss once it faded
    // (COG-6233 UX audit).
    notifications.show({
      title: "Cognify complete",
      message: `${newborn.length} new ${newborn.length === 1 ? "entity" : "entities"} joined the model${src ? ` from ${src}` : ""}.`,
      color: "teal",
      autoClose: 5000,
    });
    // A durable log line alongside the narration — the narration fades in
    // ~9s (useNarration), so "what got added recently" was only knowable at
    // the instant it happened; this keeps a short history of it.
    constructionLogSeq.current += 1;
    setConstructionLog((prev) => [
      { id: `growth-${constructionLogSeq.current}`, text: `+${newborn.length} entities${src ? ` · ${src}` : ""}` },
      ...prev,
    ].slice(0, 30));
    if (src) {
      setFlashedSourceName(src);
      setTimeout(() => setFlashedSourceName((current) => (current === src ? null : current)), 2000);
    }
  }, [narrate]);

  const qa = useBusinessQaSurface(
    canvasRef,
    live.latestSearchEvent,
    live.consumeLatestSearchEvent,
    governanceIndex.agents,
    scene.brainState,
    setSpotlight,
    setFocusSets,
    narrate,
    scene.activeDatasetId,
  );

  const tour = useBusinessTour(canvasRef, narrate, scene.brainState);
  // Paused while a source focus is active, not just during the tour: insights
  // pick from the WHOLE graph (cross-source bridges by design), so with a
  // focus lens on they'd spotlight and narrate entities the lens is currently
  // dimming — which reads as the graph showing content from outside the
  // selected source (COG-6233).
  useBusinessAutoInsights(canvasRef, scene.brainState, narrate, setSpotlight, !tour.isPlaying && !focusSets);

  // Any pointer interaction ends the tour on the spot — waiting for the
  // script to finish (or hunting for the stop control through the blur)
  // read as the view hanging. Capture phase, so d3-zoom's own mousedown
  // handling on the canvas can't swallow the event first.
  useEffect(() => {
    if (!tour.isPlaying) return;
    const stop = () => tour.stop();
    document.addEventListener("pointerdown", stop, true);
    return () => document.removeEventListener("pointerdown", stop, true);
  }, [tour, tour.isPlaying]);

  // CLO-606: resolves the asking agent's display name for BusinessCanvas's
  // marker — attribution itself (qa.askingPrincipalId) is agents[0], a
  // literal port of playSearchEvent's own behavior (see useBusinessQaSurface).
  // Null whenever no agent is asking, or the id doesn't resolve to a real
  // governance node (a stale id from a dataset switch mid-answer).
  const asking = qa.asking;
  const askingAgent = asking
    ? (() => {
        const agent = governanceIndex.agents.find((a) => a.id === asking.principalId);
        return agent
          ? { id: agent.id, name: String(agent.name || "agent"), startedAt: asking.startedAt, until: asking.until }
          : null;
      })()
    : null;

  const docCountFor = (entity: BusinessEntity | null): number => {
    if (!entity || !scene.brainState) return 0;
    return scene.brainState.docLinks.filter((l) => l._sid === entity.id || l._tid === entity.id).length;
  };

  const connectionCountFor = (entity: BusinessEntity | null): number => {
    if (!entity || !scene.brainState) return 0;
    return scene.brainState.semanticLinks.filter((l) => l._sid === entity.id || l._tid === entity.id).length;
  };

  // Ports "the rendered dataset's sources dim too when this user can't read
  // it" (customer_tutorial.html ~6085-6088) — true (unaffected) whenever no
  // principal is hovered or there's no active dataset to check against.
  const sourcesReachable = !hoveredPrincipalId || !scene.activeDatasetId
    ? true
    : accessibleDatasetIds(governanceIndex, hoveredPrincipalId).has(scene.activeDatasetId);

  // Gates anything that only makes sense once the active dataset actually
  // has a graph — a pending live-answer chip or the reasoning-trace/spotlight
  // it would trigger can't point at anything real when there are no
  // entities or links to point at (see BusinessEmptyState below).
  const hasGraphContent = Boolean(scene.brainState?.entities.length || scene.brainState?.semanticLinks.length);
  // React Query keeps the last successful `data` through a failed background
  // refetch, so a single failed 8s poll tick sets this while a perfectly good
  // graph is still on screen — every consumer must pair it with "and there is
  // nothing to show", or the failure state covers a working canvas (BusinessEmptyState
  // is a full-bleed inset-0 overlay, so it also swallows hover/click/zoom).
  const contentFailed = scene.contentError !== null && scene.contentError !== undefined;
  // Same reasoning as contentFailed, for the OTHER fetch. The empty state
  // below reads governanceIndex.datasets, so a failed governance fetch is
  // indistinguishable from a workspace with no brains — and told a user with
  // a perfectly good workspace to go create one.
  const governanceFailed = scene.governanceError !== null && scene.governanceError !== undefined;

  // The viewing narration counts what's in the dataset you switched TO, and
  // only that dataset's own graph knows those numbers — at click time
  // brainState still describes the dataset you left. So BrainSwitcher's
  // onSelect just records the intent and this narrates once the new world has
  // landed (immediately for a cached dataset, after the fetch for a cold one).
  const [pendingViewingDatasetId, setPendingViewingDatasetId] = useState<string | null>(null);
  useEffect(() => {
    if (!pendingViewingDatasetId || pendingViewingDatasetId !== scene.activeDatasetId) return;
    // A failed graph fetch has no counts to report, and the error state below
    // says so far better than "0 entities below" would.
    if (contentFailed) {
      setPendingViewingDatasetId(null);
      return;
    }
    // The content layer's own payload, not just "not loading": brainState turns
    // non-null off the governance layer alone, so counting from it while the new
    // dataset's graph is still in flight would report zero of everything.
    if (!scene.layerData[CONTENT_LAYER_ID] || !scene.brainState) return;
    narrate(computeViewingNarration(pendingViewingDatasetId, governanceIndex, scene.brainState), "#43D9E8");
    setPendingViewingDatasetId(null);
    // governanceIndex is read but deliberately not a dependency — the gate
    // above (pendingViewingDatasetId matching the now-active dataset) is what
    // decides when this runs, not whether the index object changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingViewingDatasetId, scene.activeDatasetId, scene.layerData, scene.brainState, contentFailed, narrate]);

  const clearSourceFocus = useCallback(() => {
    setFocusSets(null);
    setSelectedSourceName(null);
    narrate("showing everything", "#7E8CA6");
    // Without this, the camera stayed wherever the focus lens had zoomed
    // it — clearing the lens then dumped the WHOLE graph back into that
    // same tight framing instead of recentering to show it properly
    // (COG-6233).
    canvasRef.current?.fit(true);
  }, [narrate]);

  const selectedSourceDetail = selectedSourceName && scene.brainState
    ? computeSourceDetail(selectedSourceName, scene.brainState)
    : null;

  // "Single point of failure" preview: what would become unreachable if
  // this entity vanished. Shared by the hub callout's hover (no camera
  // move — hovering shouldn't yank the view while exploring) and NodePanel's
  // deliberate click (flies the camera to the would-be islands).
  const runWhatIfRemoval = useCallback(
    (entityId: string, entityName: string, flyCamera: boolean) => {
      if (!scene.brainState) return;
      const result = computeWhatIfRemoval(entityId, scene.brainState.semanticLinks);
      if (!result.orphanedIds.size) {
        narrate(`removing ${entityName} wouldn't disconnect anything, no single point of failure here`, "#56DB7D");
        return;
      }
      const startedAt = performance.now();
      setSpotlight({ ids: result.orphanedIds, startedAt, until: startedAt + WHAT_IF_SPOTLIGHT_MS, source: "whatIf" });
      narrate(
        `removing ${entityName} would split the model into ${result.islandCount} disconnected pieces, ${result.orphanedIds.size} record${result.orphanedIds.size === 1 ? "" : "s"} stranded`,
        "#F5566B",
      );
      if (flyCamera) requestAnimationFrame(() => canvasRef.current?.focusOnIds(result.orphanedIds));
    },
    [scene.brainState, setSpotlight, narrate],
  );

  return (
    // text-[12px] is this view's base size, and the only way to size raw
    // <button>/<input> elements here: Mantine's element reset
    // (button/input { font: inherit }, `mantine` layer) beats every Tailwind
    // utility placed directly on a control (`tailwind` layer is declared
    // first in globals.css), so controls take whatever their nearest sized
    // ancestor says — never a text-size class of their own.
    <div
      className={`bv-root relative min-h-0 w-full flex-1 overflow-hidden text-[12px] text-[#E9EEF6]${tour.isPlaying ? " bv-tour-active" : ""}`}
      style={{
        background: "radial-gradient(1200px 700px at 50% 42%, #141D33, #0E1526)",
      }}
    >
      <BusinessCanvas
        ref={canvasRef}
        brainState={scene.brainState}
        selectedId={selection.selectedEntity?.id ?? null}
        onSelectEntity={(entity, shiftKey) => {
          setSelectedSourceName(null);
          selection.handleCanvasSelect(entity, shiftKey);
        }}
        onHover={setHoveredHit}
        onHoverMove={(x, y, containerWidth) => setHoverPos({ x, y, containerWidth })}
        onBackgroundClick={() => {
          // Only clears the clicked-entity selection. Clearing the source
          // focus lens here too meant tapping empty canvas space — easy to
          // do by accident while exploring — silently revealed entities
          // from sources the user had deliberately filtered out; that lens
          // has its own toggle (click the active source card again).
          selection.clearSelection();
          // The auto-insight spotlight (useBusinessAutoInsights) and the
          // what-if removal spotlight both carry their own `until` timer
          // (9s / WHAT_IF_SPOTLIGHT_MS) and used to keep dimming/highlighting
          // the rest of the canvas for however long was left on that timer,
          // even after the user clicked empty space specifically to dismiss
          // it — clicking looked unresponsive for up to several seconds.
          // Background click is an explicit "stop showing me this" gesture,
          // same as the hub-hover-off and dataset-switch cases below that
          // already clear it immediately.
          setSpotlight(null);
        }}
        spotlight={spotlight}
        focusSets={focusSets}
        onLevelChange={(level, plumbing) => setAltimeterState({ level, plumbing })}
        sourceCardRefs={sourceCardRefs}
        activeDatasetId={scene.activeDatasetId}
        onGrowth={handleGrowth}
        answeredIds={live.answeredIds}
        pathIds={selection.pathIds}
        pathEdgeKeys={selection.pathEdgeKeys}
        askingAgent={askingAgent}
        hoveredPrincipalId={hoveredPrincipalId ?? hoveredMarkerAgentId}
        onHoverAgent={setHoveredMarkerAgentId}
      />
      <HoverTooltip
        hit={hoveredHit}
        docCount={docCountFor(hoveredHit?.kind === "entity" ? hoveredHit.node : null)}
        connectionCount={connectionCountFor(hoveredHit?.kind === "entity" ? hoveredHit.node : null)}
        x={hoverPos.x}
        y={hoverPos.y}
        containerWidth={hoverPos.containerWidth}
      />
      <SearchBar
        cogniInstance={cogniInstance}
        activeDatasetId={scene.activeDatasetId}
        onAnswer={qa.showManualAnswer}
        entities={scene.brainState?.entities ?? null}
        onPickEntity={(entity) => {
          setSelectedSourceName(null);
          selection.selectEntity(entity);
          requestAnimationFrame(() => canvasRef.current?.focusOnIds(new Set([entity.id])));
        }}
      />
      <BusinessLegend />
      <GraphConstructionLog entries={constructionLog} />
      {/* leading-[15px] pins the label's line box so the switcher chip's top
          lands at exactly 10+15+4 = 29px — SearchBar (top-[29px]) and
          OperatorsRail's workspace card align to that same line. */}
      <div className="absolute left-2.5 top-2.5 z-10">
        <div className="mb-1 px-1 text-[10px] leading-[15px] uppercase tracking-widest text-[#7E8CA6]">brain</div>
        <BrainSwitcher
          brains={brainsQuery.data ?? null}
          index={governanceIndex}
          activeDatasetId={scene.activeDatasetId}
          onSelect={(id) => {
            scene.setActiveDatasetId(id);
            selection.clearSelection();
            // A focus lens (source click or session-memory) from the PREVIOUS
            // dataset otherwise survives the switch: none of its names match
            // anything in the new dataset, so every entity dims and the
            // canvas reads as blank with no obvious way back (COG-6233).
            setFocusSets(null);
            setSelectedSourceName(null);
            setSpotlight(null);
            setConstructionLog([]);
            // Narrated once the new dataset's own graph lands, not here — see
            // pendingViewingDatasetId.
            setPendingViewingDatasetId(id);
          }}
          hoveredPrincipalId={hoveredPrincipalId}
        />
      </div>
      {/* top offset must clear the workspace label + BrainSwitcher block
          above (left-2.5 top-2.5) so it doesn't cover this panel's own
          "sources" heading. bottom offset leaves room for
          GraphConstructionLog (h-[104px], anchored bottom-24) below it in
          the same left column (COG-6233). */}
      <div className="absolute left-0 top-[68px] bottom-[208px] w-[196px]">
        <ScrollFadeContainer className="h-full overflow-y-auto">
          <SourcesRail
            brainState={scene.brainState}
            focusSets={focusSets}
            onToggleFocus={(name) => {
              const clearing = Boolean(focusSets && focusSets.has(name) && focusSets.size === 1);
              if (clearing) {
                clearSourceFocus();
                return;
              }
              setSelectedSourceName(name);
              const entityCount = scene.brainState?.setEntityCount[name] ?? 0;
              // A source with zero extracted entities (e.g. Skill/other-stage
              // content) has nothing an entity-keyed focus lens can show —
              // entering the lens would dim every entity in the graph and
              // read as the canvas going blank, with no visible content and
              // no obvious reason why (COG-6233).
              if (!entityCount) {
                const memberCount = scene.brainState?.setMemberCount[name] ?? 0;
                narrate(
                  `${sourceLabel(name)} has no extracted entities yet — ${memberCount} item${memberCount === 1 ? "" : "s"} not shown as a graph`,
                  "#7E8CA6",
                );
                return;
              }
              setFocusSets(new Set([name]));
              narrate(`showing only ${sourceLabel(name)} — ${entityCount} entities · click again for everything`, "#43D9E8");
              // Without this, clicking a source only dimmed everyone else —
              // invisible if the camera happened to be looking somewhere else
              // already, which read as the click doing nothing at all.
              const ids = new Set(
                (scene.brainState?.entities ?? []).filter((e) => setsOf(e).includes(name)).map((e) => e.id),
              );
              if (ids.size) requestAnimationFrame(() => canvasRef.current?.focusOnIds(ids));
            }}
            registerCardRef={registerSourceCardRef}
            flashSourceName={flashedSourceName}
            reachableByHoveredPrincipal={sourcesReachable}
          />
        </ScrollFadeContainer>
      </div>
      {/* Docked above the legend rather than floating near the search bar:
          an arriving answer is a passing notice, not something that should
          contest the primary input. Clicking replays the reasoning-trace
          walk; the AnswerCard then surfaces on its own via finalizeAnswer.
          Hidden while an AnswerCard is up — both dock at the same spot, and
          a live event landing during the card's ~14s lifetime would render
          under it, burying the notice's dismiss and play controls. The
          pending event survives in state, so the notice surfaces once the
          card expires or is dismissed. */}
      {qa.pendingSearchEvent && !qa.answerEvent && hasGraphContent && (
        <div
          className="absolute bottom-12 left-3 z-10 max-w-[260px] rounded-lg border px-3 py-2"
          style={{ background: "rgba(26,36,56,.92)", borderColor: "rgba(245,168,60,.4)" }}
        >
          <button
            type="button"
            onClick={qa.dismissPendingSearchEvent}
            aria-label="dismiss"
            className="absolute right-2 top-1.5 text-[#7E8CA6] hover:text-[#E9EEF6]"
          >
            ✕
          </button>
          <div className="pr-4 text-[12px] font-semibold text-[#F5A83C]">this graph just answered a question</div>
          <div className="mt-0.5 truncate text-[11.5px] text-[#7E8CA6]">
            &ldquo;{truncate(String(qa.pendingSearchEvent.question || ""), 60)}&rdquo;
          </div>
          <button
            type="button"
            onClick={qa.playPendingSearchEvent}
            className="mt-1.5 cursor-pointer font-medium text-[#43D9E8] hover:underline"
          >
            ▶ see what it used
          </button>
        </div>
      )}
      <AnswerCard event={qa.answerEvent} onDismiss={qa.dismissAnswer} />
      <SessionMemoryCard
        principalName={qa.sessionMemoryPrincipalName}
        events={live.events}
        distilledSets={qa.distilledSets}
        onDismiss={qa.dismissSessionMemory}
      />
      <OperatorsRail
        index={governanceIndex}
        focusedDatasetId={scene.activeDatasetId}
        askingPrincipalId={qa.askingPrincipalId}
        onHoverPrincipal={setHoveredPrincipalId}
        onOpenSessionMemory={(principalId, principalName) => {
          setSelectedSourceName(null);
          qa.onOpenSessionMemory(principalId, principalName);
        }}
      />
      {selection.selectedEntity ? (
        <NodePanel
          entity={selection.selectedEntity}
          docCount={docCountFor(selection.selectedEntity)}
          connectionCount={connectionCountFor(selection.selectedEntity)}
          events={live.events}
          onClose={selection.clearSelection}
          pathTargetName={selection.pathTargetName}
          pathHops={selection.pathHops}
          onClearPath={selection.clearPath}
          onSimulateRemoval={() => {
            if (!selection.selectedEntity) return;
            runWhatIfRemoval(selection.selectedEntity.id, String(selection.selectedEntity.name || "this record"), true);
          }}
        />
      ) : (
        <SourceDetailCard detail={selectedSourceDetail} onClose={clearSourceFocus} />
      )}
      <BusinessDock
        narration={narration}
        altimeter={altimeterState}
        onAltimeterLevel={(level) => canvasRef.current?.goToAltimeterLevel(level)}
        live={live.live}
        tourPlaying={tour.isPlaying}
        onTourStart={tour.start}
        onTourStop={tour.stop}
        recordCount={scene.brainState?.plumbingNodes.length ?? 0}
      />
      {/* Only the true "content not here yet" states: no brainState at all,
          or the focused dataset's own graph still fetching (first visit to a
          dataset — a cached switch skips this entirely and presents
          pre-settled via useBusinessSimulation's settleKey). Deliberately
          NOT scene.isLoading: a background layer query (governance) hanging
          must never pin a loading overlay over a rendered graph. Also off once
          the graph fetch has failed: with both layer queries failing brainState
          stays null forever, which otherwise pinned this "weaving…" overlay
          under the error state below with nothing left to wait for. */}
      {scene.activeDatasetId && (!scene.brainState || scene.isContentLoading) && !contentFailed && (
        <BusinessLoading label="weaving your business model…" />
      )}
      {/* No dataset selected is its own state, not an empty dataset — the
          auto-focus effect above normally picks the first brain, so this
          only shows when there is genuinely nothing to pick. Gated on the
          governance list actually being empty, the same source auto-focus
          reads: keying it to /visualize/brains instead meant a workspace
          whose datasets were all present but whose heavy brains fetch had
          rejected was told it had no data at all. Also gated on the list
          being empty rather than on activeDatasetId alone, because
          auto-focus runs in a post-paint effect — for one paint after
          governance resolves activeDatasetId is still null. */}
      {!scene.isLoading && !scene.activeDatasetId && governanceIndex.datasets.length === 0 && !governanceFailed && (
        <BusinessEmptyState label="no dataset selected — create a brain and upload documents to see your business model" />
      )}
      {!scene.isLoading && !scene.activeDatasetId && governanceFailed && (
        <BusinessEmptyState label="couldn't load this workspace's brains — check your connection and reload" />
      )}
      {/* A failed graph fetch must never masquerade as "this dataset is
          empty" — without this branch a /visualize/json 500 rendered the
          cheerful no-content state below. Only when there's genuinely nothing
          on screen though: a graph that's already rendered keeps rendering
          through a failed poll tick (see contentFailed), and covering it with
          this overlay blocked every interaction until a later tick succeeded. */}
      {scene.activeDatasetId && contentFailed && !hasGraphContent && (
        <BusinessEmptyState label="couldn't load this dataset's graph — check your connection and try switching to it again" />
      )}
      {/* The filament that used to draw a stray line to an empty source's
          territory is gated on this same "no entities, no links" condition
          (see BusinessCanvas.tsx) — this overlay is what replaces it. */}
      {!scene.isLoading && scene.activeDatasetId && !contentFailed && scene.brainState && !hasGraphContent && (
        <BusinessEmptyState
          label={
            scene.brainState.sourceNames.length
              ? "content ingested, nothing extracted into the graph yet. check back after processing finishes"
              : "no content in this dataset yet"
          }
        />
      )}
    </div>
  );
}

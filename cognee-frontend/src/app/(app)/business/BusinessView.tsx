"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { notifications } from "@mantine/notifications";
import type { CogneeInstance } from "@/modules/instances/types";
import { useStructuredExplorer, type RowFilter } from "@/modules/business/useStructuredExplorer";
import StructuredExplorerPanel from "@/modules/business/panels/StructuredExplorerPanel";
import { useBusinessScene } from "@/modules/business/useBusinessScene";
import { useBusinessColorMode } from "@/modules/business/useBusinessColorMode";
import { EMPTY_BRAIN_STATE } from "@/modules/business/canvas/businessCanvasHelpers";
import { useBrainsSummary } from "@/modules/business/useBrainsSummary";
import { useGovernanceIndex } from "@/modules/business/useGovernanceIndex";
import { useBusinessLiveUpdates } from "@/modules/business/useBusinessLiveUpdates";
import { computeViewingNarration } from "@/modules/business/viewingNarration";
import type { BusinessCanvasHandle } from "@/modules/business/canvas/BusinessCanvas";
import BusinessGraphCanvas from "@/modules/business/canvas/BusinessGraphCanvas";
import type { Spotlight } from "@/modules/business/canvas/businessDraw";
import type { BusinessEntity } from "@/modules/business/sceneTypes";
import NodePanel from "@/modules/business/panels/NodePanel";
import SourceDetailCard from "@/modules/business/panels/SourceDetailCard";
import { computeSourceDetail } from "@/modules/business/computeSourceDetail";
import { computeWhatIfRemoval } from "@/modules/business/computeWhatIfRemoval";
import OperatorsRail from "@/modules/business/panels/OperatorsRail";
import BrainSwitcher from "@/modules/business/panels/BrainSwitcher";
import SourcesRail from "@/modules/business/panels/SourcesRail";
import TypesRail from "@/modules/business/panels/TypesRail";
import GraphConstructionLog, { type ConstructionLogEntry } from "@/modules/business/panels/GraphConstructionLog";
import AnswerCard from "@/modules/business/panels/AnswerCard";
import SessionMemoryCard from "@/modules/business/panels/SessionMemoryCard";
import { GOVERNANCE_LAYER_ID } from "@/modules/business/layers/governanceLayer";
import { CONTENT_LAYER_ID } from "@/modules/business/layers/contentLayer";
import { accessibleDatasetIds, narrowToReadable } from "@/modules/business/useGovernanceIndex";
import noDatasetStateLabel from "@/modules/business/noDatasetStateLabel";
import { countLinksTouching, setsOf, sourceLabel } from "@/modules/business/computeBrainState";
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
import useSeedDatasetId from "@/modules/business/useSeedDatasetId";
import { useFilter } from "@/ui/layout/FilterContext";
import { useEntitySelection } from "@/modules/business/useEntitySelection";

// Hidden for now: the explorer is built but not ready to ship in the Business view.
const SHOW_STRUCTURED_EXPLORER = false;

const WHAT_IF_SPOTLIGHT_MS = 12000;

interface BusinessViewProps {
  cogniInstance: CogneeInstance;
}

// Requires a ready cogniInstance so every hook below can assume it's real —
// CustomAppShell already shows the pod-provisioning state for "/business"
// before this ever mounts (see POD_DEPENDENT_PATHS), the same gate every
// other pod-backed route uses.
export default function BusinessView({ cogniInstance }: BusinessViewProps) {
  // Held here because the two hooks below need it in opposite directions:
  // the scene's content layer decides whether to poll for graph growth from
  // it, and the live-updates hook - which is what learns it - needs the
  // scene's focused dataset to open its socket at all. One piece of state
  // between them breaks that cycle; it starts false so the graph polls until
  // a socket handshake actually completes.
  const [graphPushActive, setGraphPushActive] = useState(false);
  const scene = useBusinessScene(cogniInstance, { graphPushActive });
  const [colorMode, setColorMode] = useBusinessColorMode();
  const encoding = (scene.brainState ?? EMPTY_BRAIN_STATE).encodings[colorMode];
  const brainsQuery = useBrainsSummary(cogniInstance);
  // Tenant-wide, straight off the provenance payload. Nothing may RENDER from
  // this one: it exists for the seed fallback below, which needs a dataset id
  // to open when the read-scoped list could not be fetched at all, and an id
  // the pod then refuses is a failed fetch rather than a disclosed name.
  const tenantGovernanceIndex = useGovernanceIndex(scene.layerData[GOVERNANCE_LAYER_ID]);
  const { display: narration, narrate } = useNarration();
  const live = useBusinessLiveUpdates(scene.activeDatasetId, cogniInstance, {
    onGraphPushChange: setGraphPushActive,
  });

  const canvasRef = useRef<BusinessCanvasHandle>(null);
  const structured = useStructuredExplorer(scene.brainState, scene.activeDatasetId, SHOW_STRUCTURED_EXPLORER);
  const selection = useEntitySelection(structured.view?.graphLinks, structured.view?.entityById);
  const [hoveredPrincipalId, setHoveredPrincipalId] = useState<string | null>(null);
  // Kept apart from hoveredPrincipalId on purpose. The rail's hover is an
  // access question ("what can this principal read"), so it dims the sources
  // rail; hovering the canvas marker is just "which agent is that", and
  // feeding it into the same state greyed the entire sources rail out from
  // under the user for pointing at a dot.
  const [hoveredMarkerAgentId, setHoveredMarkerAgentId] = useState<string | null>(null);
  const [focusSets, setFocusSets] = useState<Set<string> | null>(null);
  const [focusTypes, setFocusTypes] = useState<Set<string> | null>(null);
  const [railTab, setRailTab] = useState<"sources" | "types">("sources");
  const [spotlight, setSpotlight] = useState<Spotlight | null>(null);
  const { setFilter: setRowFilter } = structured;
  const { clearSelection } = selection;
  const changeRowFilter = useCallback((filter: RowFilter) => {
    setRowFilter(filter);
    clearSelection();
    setSpotlight(null);
  }, [setRowFilter, clearSelection]);
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
  // Seeded from the dataset list, not from governance, for two reasons.
  // Correctness: `/v1/schema/provenance/json` scopes datasets by tenant
  // (Dataset.tenant_id), so its first entry can be one this caller holds no
  // read grant on, and the graph fetch for it then fails; `/v1/datasets/`
  // answers with exactly the readable set (get_all_user_permission_datasets)
  // the graph endpoint will accept. Speed: that list is already in flight
  // from the app layout's FilterProvider, which never remounts, so the graph
  // fetch starts alongside governance instead of behind it — governance is a
  // whole-tenant provenance graph the canvas does not need to pick a dataset.
  const { selectedDataset, datasets, datasetsUnavailable, loading: datasetsLoading } = useFilter();
  const readableDatasetIds = useMemo(() => datasets.map((d) => d.id), [datasets]);
  const governanceDatasetIds = useMemo(
    () => tenantGovernanceIndex.datasets.map((d) => d.id),
    [tenantGovernanceIndex.datasets],
  );
  // What every panel below is allowed to see. Narrowed once, here, so no
  // panel has to remember to filter — and narrowed to NOTHING while the
  // readable list is still answering or has failed, because an unknown grant
  // is not a grant (CLO-932 follow-up: the rails and the switcher were
  // naming other members' datasets).
  const governanceIndex = useMemo(
    () => narrowToReadable(
      tenantGovernanceIndex,
      datasetsUnavailable || datasetsLoading ? null : readableDatasetIds,
    ),
    [tenantGovernanceIndex, datasetsUnavailable, datasetsLoading, readableDatasetIds],
  );
  const brainListState = datasetsLoading
    ? "loading" as const
    : datasetsUnavailable
      ? "unavailable" as const
      : "ready" as const;
  const { unconfirmedSeedId } = useSeedDatasetId({
    instanceId: cogniInstance.instanceId,
    activeDatasetId: scene.activeDatasetId,
    setActiveDatasetId: scene.setActiveDatasetId,
    selectedDatasetId: selectedDataset?.id ?? null,
    readableDatasetIds,
    // `loading` is isLoading, which reads false for a DISABLED query - but
    // that query is gated on exactly the three conditions CustomAppShell
    // gates this route on (cogniInstance, !isInitializing, tenantReady;
    // "/business" is in POD_DEPENDENT_PATHS and podPending renders the
    // provisioning screen instead of children), so it is never disabled while
    // this view is mounted.
    readableLoading: datasetsLoading,
    readableUnavailable: datasetsUnavailable,
    governanceDatasetIds,
  });

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
      `this is your business — ${brain.typeNodes.length} kind${brain.typeNodes.length === 1 ? "" : "s"} of things across ${brain.sourceNames.length} source${brain.sourceNames.length === 1 ? "" : "s"} in this graph`,
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
  useBusinessAutoInsights(canvasRef, scene.brainState, narrate, setSpotlight, !tour.isPlaying && !focusSets && !focusTypes && structured.rowIds === null);

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
    return countLinksTouching(scene.brainState.docLinks, entity.id);
  };

  const connectionCountFor = (entity: BusinessEntity | null): number => {
    if (!entity || !scene.brainState) return 0;
    return countLinksTouching(scene.brainState.graphLinks, entity.id);
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
  const hasGraphContent = Boolean(scene.brainState?.entities.length || scene.brainState?.graphLinks.length);
  // React Query keeps the last successful `data` through a failed background
  // refetch, so a single failed 8s poll tick sets this while a perfectly good
  // graph is still on screen — every consumer must pair it with "and there is
  // nothing to show", or the failure state covers a working canvas (BusinessEmptyState
  // is a full-bleed inset-0 overlay, so it also swallows hover/click/zoom).
  // A fetch for the dataset we guessed from last visit is not evidence of
  // anything until the readable list vouches for that guess: the id may
  // simply have been deleted or unshared since. Reporting "check your
  // connection" for those seconds blames the network for our own bet, so the
  // failure only counts once the guess is confirmed (or replaced).
  const seedGuessUnconfirmed = unconfirmedSeedId !== null
    && scene.activeDatasetId === unconfirmedSeedId;
  const contentFailed = !seedGuessUnconfirmed
    && scene.contentError !== null && scene.contentError !== undefined;
  // Same reasoning as contentFailed, for the OTHER fetch. The empty state
  // below reads governanceIndex.datasets, so a failed governance fetch is
  // indistinguishable from a workspace with no brains — and told a user with
  // a perfectly good workspace to go create one.
  const governanceFailed = scene.governanceError !== null && scene.governanceError !== undefined;

  const noDatasetLabel = noDatasetStateLabel({
    isLoading: scene.isLoading,
    activeDatasetId: scene.activeDatasetId,
    listState: brainListState,
    readableDatasetCount: governanceIndex.datasets.length,
    governanceFailed,
  });

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
    ? computeSourceDetail(selectedSourceName, scene.brainState, encoding)
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
      <BusinessGraphCanvas
        ref={canvasRef}
        brainState={scene.brainState}
        viewState={structured.view}
        rowFilterKey={structured.filterKey}
        colorMode={colorMode}
        selectedId={selection.selectedEntity?.id ?? null}
        onSelectEntity={(entity, shiftKey) => {
          setSelectedSourceName(null);
          selection.handleCanvasSelect(entity, shiftKey);
        }}
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
        focusTypes={focusTypes}
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
      <SearchBar
        cogniInstance={cogniInstance}
        activeDatasetId={scene.activeDatasetId}
        onAnswer={qa.showManualAnswer}
        entities={structured.view?.entities ?? null}
        onPickEntity={(entity) => {
          setSelectedSourceName(null);
          selection.selectEntity(entity);
          requestAnimationFrame(() => canvasRef.current?.focusOnIds(new Set([entity.id])));
        }}
      />
      {SHOW_STRUCTURED_EXPLORER && (
        <StructuredExplorerPanel index={structured.index} filter={structured.filter}
          onChange={changeRowFilter} matchingCount={structured.rowIds?.size ?? null} />
      )}
      <BusinessLegend colorMode={colorMode} onColorModeChange={setColorMode} />
      <GraphConstructionLog entries={constructionLog} />
      {/* leading-[15px] pins the label's line box so the switcher chip's top
          lands at exactly 10+15+4 = 29px — SearchBar (top-[29px]) and
          OperatorsRail's workspace card align to that same line. */}
      <div className="absolute left-2.5 top-2.5 z-10">
        <div className="mb-1 px-1 text-[10px] leading-[15px] uppercase tracking-widest text-[#7E8CA6]">dataset</div>
        {/* The index is already narrowed to the readable set; listState is
            what lets the dropdown say WHY it is empty — still loading, the
            list failed, or this caller may genuinely read nothing. */}
        <BrainSwitcher
          brains={brainsQuery.data ?? null}
          index={governanceIndex}
          listState={brainListState}
          activeDatasetId={scene.activeDatasetId}
          onSelect={(id) => {
            scene.setActiveDatasetId(id);
            selection.clearSelection();
            // A focus lens (source click or session-memory) from the PREVIOUS
            // dataset otherwise survives the switch: none of its names match
            // anything in the new dataset, so every entity dims and the
            // canvas reads as blank with no obvious way back (COG-6233).
            setFocusSets(null);
            setFocusTypes(null);
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
        {/* CLO-660: a type-keyed lens sits beside the source one, in the same
            rail — reuses the exact card UI, no new screen space. Switching
            tabs keeps the other lens's selection active in the background
            (the dot marks a hidden filter still applied); inLens composes
            both when both are set. */}
        <div className="flex gap-1 px-1 pt-1">
          <button
            type="button"
            onClick={() => setRailTab("sources")}
            className={`flex-1 border-b-2 pb-1.5 text-[10px] uppercase tracking-widest ${
              railTab === "sources" ? "border-[#43D9E8] text-[#E9EEF6]" : "border-transparent text-[#7E8CA6]"
            }`}
          >
            sources
            {railTab !== "sources" && focusSets && <span className="ml-1 inline-block h-1.5 w-1.5 rounded-full bg-[#43D9E8]" />}
          </button>
          <button
            type="button"
            onClick={() => setRailTab("types")}
            className={`flex-1 border-b-2 pb-1.5 text-[10px] uppercase tracking-widest ${
              railTab === "types" ? "border-[#43D9E8] text-[#E9EEF6]" : "border-transparent text-[#7E8CA6]"
            }`}
          >
            types
            {railTab !== "types" && focusTypes && <span className="ml-1 inline-block h-1.5 w-1.5 rounded-full bg-[#43D9E8]" />}
          </button>
        </div>
        <ScrollFadeContainer className="h-[calc(100%-32px)] overflow-y-auto">
          {railTab === "sources" ? (
            <SourcesRail
              brainState={scene.brainState}
              encoding={encoding}
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
          ) : (
            <TypesRail
              typeNodes={scene.brainState?.typeNodes ?? null}
              encoding={encoding}
              focusTypes={focusTypes}
              onToggleFocus={(name) => {
                const clearing = Boolean(focusTypes && focusTypes.has(name) && focusTypes.size === 1);
                if (clearing) {
                  setFocusTypes(null);
                  narrate("showing everything again", "#7E8CA6");
                  return;
                }
                const typeNode = scene.brainState?.typeNodes.find((tn) => tn.name === name);
                const count = typeNode?.members.length ?? 0;
                setFocusTypes(new Set([name]));
                narrate(`isolating ${count} ${name} node${count === 1 ? "" : "s"} · click again for everything`, "#43D9E8");
                // Mirrors the sources handler above: without this, clicking a
                // type only dimmed everyone else — invisible if the camera was
                // looking elsewhere already. Worse here than for sources, since
                // a type lens HIDES non-matching entities rather than dimming
                // them, so the viewport could go fully empty instead of dim.
                const ids = new Set((typeNode?.members ?? []).map((m) => m.id));
                if (ids.size) requestAnimationFrame(() => canvasRef.current?.focusOnIds(ids));
              }}
            />
          )}
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
        listState={brainListState}
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
          auto-focus effect above normally picks the first brain, so this only
          shows when there is genuinely nothing to pick. Gated on the list
          being empty rather than on activeDatasetId alone, because auto-focus
          runs in a post-paint effect: for one paint after the list resolves,
          activeDatasetId is still null. Which of these messages is owed, and
          when none is, lives in noDatasetStateLabel — see its tests for the
          combinations. */}
      {noDatasetLabel && <BusinessEmptyState label={noDatasetLabel} />}
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

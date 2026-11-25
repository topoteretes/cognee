"use client";

import { forwardRef, memo, useCallback, useState, type ComponentProps } from "react";
import { countLinksTouching } from "@/modules/business/computeBrainState";
import HoverTooltip from "@/modules/business/panels/HoverTooltip";
import BusinessCanvas, { type BusinessCanvasHandle } from "./BusinessCanvas";
import type { SceneHit } from "./businessHitTest";

type Props = Omit<ComponentProps<typeof BusinessCanvas>, "ref" | "onHover" | "onHoverMove">;
const MemoCanvas = memo(BusinessCanvas);

// Cursor movement belongs to the tooltip. Keeping it here prevents every
// mousemove from rerendering BusinessView and rerunning its panels' effects.
const BusinessGraphCanvas = forwardRef<BusinessCanvasHandle, Props>(function BusinessGraphCanvas(props, ref) {
  const [hover, setHover] = useState<{ hit: SceneHit; datasetId: Props["activeDatasetId"] } | null>(null);
  const [position, setPosition] = useState({ x: 0, y: 0, containerWidth: 0 });
  const { brainState, viewState = null, activeDatasetId } = props;
  const visibleBrain = viewState ?? brainState;
  const onHover = useCallback((hit: SceneHit) => {
    setHover((previous) => previous?.hit === hit && previous.datasetId === activeDatasetId
      ? previous : { hit, datasetId: activeDatasetId });
  }, [activeDatasetId]);
  const onHoverMove = useCallback((x: number, y: number, containerWidth: number) => {
    setPosition((previous) => previous.x === x && previous.y === y && previous.containerWidth === containerWidth
      ? previous : { x, y, containerWidth });
  }, []);

  const hovered = hover?.datasetId === activeDatasetId ? hover?.hit ?? null : null;
  const filtered = visibleBrain !== brainState;
  const hit = hovered?.kind === "entity" && !visibleBrain?.entityById[hovered.node.id]
    ? null : filtered && hovered?.kind === "type" ? null : hovered;
  const entityId = hit?.kind === "entity" ? hit.node.id : null;
  // Counts are facts about the data, not the filtered view, so they match the
  // node panel's.
  const docCount = entityId && brainState ? countLinksTouching(brainState.docLinks, entityId) : 0;
  const connectionCount = entityId && brainState ? countLinksTouching(brainState.graphLinks, entityId) : 0;

  return (
    <>
      <MemoCanvas {...props} ref={ref} onHover={onHover} onHoverMove={onHoverMove} />
      <HoverTooltip hit={hit} docCount={docCount} connectionCount={connectionCount} {...position} />
    </>
  );
});

export default BusinessGraphCanvas;

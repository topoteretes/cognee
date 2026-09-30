// Standalone entry for the Business canvas, served by the Python backend's
// `visualize` endpoint and `cognee.visualize_graph()`.
//
// It mounts the same BusinessCanvas / computeBrainState the Business page
// uses, but reads the graph from `window.__COGNEE_PAYLOAD__` (the
// GET /v1/visualize/json shape, embedded into the page by
// cognee/modules/visualization/business_visualization.py) instead of
// fetching it. `npm run build:standalone` bundles this file, with React and
// d3 included, into cognee/modules/visualization/views/business_standalone.js.
import { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import BusinessCanvas, { type BusinessCanvasHandle } from "@/modules/business/canvas/BusinessCanvas";
import { computeBrainState } from "@/modules/business/computeBrainState";
import type { SceneHit } from "@/modules/business/canvas/businessHitTest";
import type { VisualizationPayload } from "@/modules/business/types";

declare global {
  interface Window {
    __COGNEE_PAYLOAD__: VisualizationPayload & { dataset_name?: string };
  }
}

function StandaloneApp() {
  const payload = window.__COGNEE_PAYLOAD__;
  const brainState = useMemo(
    () => computeBrainState(payload.nodes, payload.links, payload.color_maps?.node_set ?? {}),
    [payload],
  );
  const canvasRef = useRef<BusinessCanvasHandle | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hover, setHover] = useState<SceneHit>(null);
  const [cursor, setCursor] = useState({ x: 0, y: 0 });
  const [level, setLevel] = useState(0);

  const selected = selectedId ? brainState.entityById[selectedId] : null;
  const hovered = hover?.kind === "entity" ? hover.node : null;
  const edgeCount = selected
    ? payload.links.filter((l) => l.source === selected.id || l.target === selected.id).length
    : 0;

  return (
    <div className="bv-root">
      <BusinessCanvas
        ref={canvasRef}
        brainState={brainState}
        selectedId={selectedId}
        onSelectEntity={(entity) => setSelectedId(entity.id)}
        onHover={setHover}
        onHoverMove={(x, y) => setCursor({ x, y })}
        onBackgroundClick={() => setSelectedId(null)}
        onLevelChange={setLevel}
        spotlight={null}
        focusSets={null}
      />
      <div className="bv-hud">
        <b>{payload.dataset_name || "graph"}</b> · {brainState.entities.length} entities ·{" "}
        {brainState.typeNodes.length} types · level {level}
        <button type="button" onClick={() => canvasRef.current?.fit(true)}>fit</button>
      </div>
      {hovered && (
        <div className="bv-tip" style={{ left: cursor.x + 14, top: cursor.y + 14 }}>
          <b>{hovered.name ?? hovered.id}</b>
          <div>
            {hovered.type ?? "—"}
            {hovered.source_node_set ? ` · ${hovered.source_node_set}` : ""}
          </div>
        </div>
      )}
      {selected && (
        <div className="bv-panel">
          <b>{selected.name ?? selected.id}</b>
          <div className="bv-muted">{selected.type} · stage {String(selected.stage)}</div>
          {typeof selected.description === "string" && <p>{selected.description}</p>}
          <div className="bv-muted">{edgeCount} edges</div>
        </div>
      )}
    </div>
  );
}

const root = document.getElementById("root");
if (!root) throw new Error("business-standalone: #root element missing");
createRoot(root).render(<StandaloneApp />);

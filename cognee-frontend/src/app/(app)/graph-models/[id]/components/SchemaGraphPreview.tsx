"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Flex, Stack, Title, UnstyledButton } from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";
import type { GraphSchema, RelationField } from "@/modules/graphModels/types";

// ── Types ─────────────────────────────────────────────────────────────────────

interface SimNode {
  id: string;
  label: string;
  type: "entity" | "missing";
  entityId: string | null;
  x: number;
  y: number;
  vx: number;
  vy: number;
  fx: number | null;
  fy: number | null;
}

interface SimLink {
  source: string;
  target: string;
  label: string;
  isMissing: boolean;
  fieldId: string;
  sourceEntityId: string;
}

interface SchemaGraphPreviewProps {
  schema: GraphSchema;
  selectedEntityId: string | null;
  onEntitySelect: (entityId: string) => void;
  onJumpToField: (entityId: string, fieldId: string) => void;
}

// ── Graph data builder ────────────────────────────────────────────────────────

function buildGraphData(
  schema: GraphSchema,
  width: number,
  height: number,
): { nodes: SimNode[]; links: SimLink[] } {
  const entityMap = new Map(schema.entities.map((e) => [e.name, e._id]));
  const cx = width / 2;
  const cy = height / 2;
  const count = schema.entities.length;
  const radius = Math.min(width, height) * 0.3;

  const nodes: SimNode[] = schema.entities.map((e, i) => {
    const angle = (2 * Math.PI * i) / Math.max(count, 1) - Math.PI / 2;
    return {
      id: e._id,
      label: e.name,
      type: "entity" as const,
      entityId: e._id,
      x: count === 1 ? cx : cx + radius * Math.cos(angle),
      y: count === 1 ? cy : cy + radius * Math.sin(angle),
      vx: 0,
      vy: 0,
      fx: null,
      fy: null,
    };
  });

  const missingNames = new Set<string>();
  const links: SimLink[] = [];

  for (const entity of schema.entities) {
    for (const field of entity.fields) {
      if (field.kind !== "relation") continue;
      const rel = field as RelationField;
      const target = rel.relation.targetEntityName;
      if (!target) continue;

      const targetEntityId = entityMap.get(target);
      const isMissing = !targetEntityId;

      if (isMissing && !missingNames.has(target)) {
        missingNames.add(target);
        const angle = Math.random() * 2 * Math.PI;
        nodes.push({
          id: `__missing__${target}`,
          label: target,
          type: "missing" as const,
          entityId: null,
          x: cx + (radius * 1.3) * Math.cos(angle),
          y: cy + (radius * 1.3) * Math.sin(angle),
          vx: 0,
          vy: 0,
          fx: null,
          fy: null,
        });
      }

      links.push({
        source: entity._id,
        target: targetEntityId ?? `__missing__${target}`,
        label: `${field.name} (${rel.relation.cardinality})`,
        isMissing,
        fieldId: field._id,
        sourceEntityId: entity._id,
      });
    }
  }

  return { nodes, links };
}

// ── Force simulation ──────────────────────────────────────────────────────────

const REPULSION = 5000;
const SPRING_LEN = 150;
const SPRING_K = 0.035;
const CENTER_K = 0.006;
const DAMPING = 0.76;

function tickOnce(
  nodes: SimNode[],
  links: SimLink[],
  width: number,
  height: number,
): number {
  // Repulsion
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[j].x - nodes[i].x || 0.01;
      const dy = nodes[j].y - nodes[i].y || 0.01;
      const dist2 = dx * dx + dy * dy;
      const dist = Math.sqrt(dist2) || 1;
      const f = REPULSION / dist2;
      const fx = (dx / dist) * f;
      const fy = (dy / dist) * f;
      nodes[i].vx -= fx;
      nodes[i].vy -= fy;
      nodes[j].vx += fx;
      nodes[j].vy += fy;
    }
  }

  // Spring along links
  for (const link of links) {
    const src = nodes.find((n) => n.id === link.source);
    const tgt = nodes.find((n) => n.id === link.target);
    if (!src || !tgt) continue;
    const dx = tgt.x - src.x;
    const dy = tgt.y - src.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const f = SPRING_K * (dist - SPRING_LEN);
    const fx = (dx / dist) * f;
    const fy = (dy / dist) * f;
    src.vx += fx;
    src.vy += fy;
    tgt.vx -= fx;
    tgt.vy -= fy;
  }

  // Center gravity
  const cx = width / 2;
  const cy = height / 2;
  for (const n of nodes) {
    n.vx += (cx - n.x) * CENTER_K;
    n.vy += (cy - n.y) * CENTER_K;
  }

  // Integrate
  let energy = 0;
  for (const n of nodes) {
    if (n.fx !== null) {
      n.x = n.fx;
      n.vx = 0;
    } else {
      n.vx *= DAMPING;
      n.x += n.vx;
    }
    if (n.fy !== null) {
      n.y = n.fy;
      n.vy = 0;
    } else {
      n.vy *= DAMPING;
      n.y += n.vy;
    }
    energy += Math.abs(n.vx) + Math.abs(n.vy);
  }

  return energy;
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function ExpandIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 3 21 3 21 9" />
      <polyline points="9 21 3 21 3 15" />
      <line x1="21" y1="3" x2="14" y2="10" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </svg>
  );
}

function CollapseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="4 14 10 14 10 20" />
      <polyline points="20 10 14 10 14 4" />
      <line x1="14" y1="10" x2="21" y2="3" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </svg>
  );
}

// ── Constants ─────────────────────────────────────────────────────────────────

const NODE_R = 13;
const ENTITY_COLOR = "#5C10F4";
const MISSING_COLOR = "#f59e0b";
const SELECTED_COLOR = "#0DFF00";

// ── Component ─────────────────────────────────────────────────────────────────

export default function SchemaGraphPreview({
  schema,
  selectedEntityId,
  onEntitySelect,
}: SchemaGraphPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 420 });
  const dimensionsRef = useRef({ width: 600, height: 420 });

  const [nodes, setNodes] = useState<SimNode[]>([]);
  const [links, setLinks] = useState<SimLink[]>([]);
  const nodesRef = useRef<SimNode[]>([]);
  const linksRef = useRef<SimLink[]>([]);

  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [containerHeight, setContainerHeight] = useState(400);
  const [manualHeight, setManualHeight] = useState<number | null>(null);

  const animFrameRef = useRef<number | null>(null);
  const dragNodeIdRef = useRef<string | null>(null);
  const didDragRef = useRef(false);
  const isDraggingResizeRef = useRef(false);

  // ── Resize observer ────────────────────────────────────────────────────────

  useEffect(() => {
    const update = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      setDimensions({ width: w, height: h });
      dimensionsRef.current = { width: w, height: h };
    };
    update();
    const ro = new ResizeObserver(update);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const update = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      setContainerHeight(Math.max(300, window.innerHeight - rect.top - 80));
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  useEffect(() => {
    if (!isExpanded) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setIsExpanded(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isExpanded]);

  // ── Simulation loop ────────────────────────────────────────────────────────

  const startSimulation = useCallback(() => {
    if (animFrameRef.current !== null) cancelAnimationFrame(animFrameRef.current);
    let frameCount = 0;

    const loop = () => {
      frameCount++;
      const { width, height } = dimensionsRef.current;
      const energy = tickOnce(nodesRef.current, linksRef.current, width, height);
      setNodes([...nodesRef.current]);
      if (energy > 0.08 && frameCount < 400) {
        animFrameRef.current = requestAnimationFrame(loop);
      } else {
        animFrameRef.current = null;
      }
    };

    animFrameRef.current = requestAnimationFrame(loop);
  }, []);

  useEffect(() => () => {
    if (animFrameRef.current !== null) cancelAnimationFrame(animFrameRef.current);
  }, []);

  // ── Rebuild on schema / dimension change ──────────────────────────────────

  useEffect(() => {
    if (dimensions.width === 0 || dimensions.height === 0) return;
    const { nodes: newNodes, links: newLinks } = buildGraphData(
      schema,
      dimensions.width,
      dimensions.height,
    );
    nodesRef.current = newNodes;
    linksRef.current = newLinks;
    setNodes([...newNodes]);
    setLinks([...newLinks]);
    startSimulation();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schema, dimensions.width > 0 && dimensions.height > 0]);

  // ── Drag ──────────────────────────────────────────────────────────────────

  function handleNodePointerDown(nodeId: string, e: React.PointerEvent) {
    e.stopPropagation();
    dragNodeIdRef.current = nodeId;
    didDragRef.current = false;
    (e.target as Element).setPointerCapture(e.pointerId);
  }

  function handleSvgPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!dragNodeIdRef.current || !svgRef.current) return;
    didDragRef.current = true;
    const rect = svgRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const node = nodesRef.current.find((n) => n.id === dragNodeIdRef.current);
    if (node) {
      node.fx = x;
      node.fy = y;
      node.x = x;
      node.y = y;
    }
    setNodes([...nodesRef.current]);
    if (animFrameRef.current === null) startSimulation();
  }

  function handleSvgPointerUp() {
    if (dragNodeIdRef.current) {
      const node = nodesRef.current.find((n) => n.id === dragNodeIdRef.current);
      if (node) { node.fx = null; node.fy = null; }
      dragNodeIdRef.current = null;
    }
  }

  // ── Drag-resize ───────────────────────────────────────────────────────────

  const handleResizeStart = useCallback((startY: number) => {
    void startY;
    isDraggingResizeRef.current = true;
    document.body.style.userSelect = "none";
    document.body.style.cursor = "row-resize";
    const onMove = (clientY: number) => {
      if (!isDraggingResizeRef.current || !containerRef.current) return;
      const top = containerRef.current.getBoundingClientRect().top;
      setManualHeight(Math.min(Math.max(200, clientY - top), window.innerHeight - 100));
    };
    const onEnd = () => {
      isDraggingResizeRef.current = false;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onEnd);
    };
    const onMouseMove = (e: MouseEvent) => onMove(e.clientY);
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onEnd);
  }, []);

  // ── Adjacency ─────────────────────────────────────────────────────────────

  const adjacentIds = hoveredNodeId
    ? new Set<string>(
        links
          .filter((l) => l.source === hoveredNodeId || l.target === hoveredNodeId)
          .flatMap((l) => [l.source, l.target])
          .concat(hoveredNodeId),
      )
    : null;

  const resolvedHeight = isExpanded
    ? undefined
    : manualHeight !== null
      ? manualHeight
      : containerHeight;

  const isEmpty = schema.entities.length === 0;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <>
      {isExpanded && (
        <div
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", zIndex: 49 }}
          onClick={() => setIsExpanded(false)}
        />
      )}

      <Stack
        className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0]"
        bg="white"
        style={
          isExpanded
            ? { position: "fixed", inset: 10, zIndex: 50, overflow: "auto", display: "flex", flexDirection: "column" }
            : undefined
        }
      >
        {/* Header */}
        <Flex justify="space-between" align="center" mb="1.625rem">
          <Title size="h2">Schema Graph</Title>
          <UnstyledButton
            onClick={() => setIsExpanded((v) => !v)}
            title={isExpanded ? "Collapse" : "Expand"}
            style={{ padding: "4px 8px", borderRadius: 4, color: tokens.textSecondary }}
          >
            {isExpanded ? <CollapseIcon /> : <ExpandIcon />}
          </UnstyledButton>
        </Flex>

        {/* Canvas area */}
        <div
          ref={containerRef}
          className="relative w-full overflow-hidden rounded-[0.375rem]"
          style={{
            height: isExpanded ? "100%" : `${resolvedHeight}px`,
            flex: isExpanded ? 1 : undefined,
            background: "linear-gradient(135deg, #101010 0%, #1a1a2e 100%)",
          }}
        >
          {isEmpty ? (
            <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: "0.75rem", color: "rgba(219,216,216,0.4)" }}>
              <p style={{ margin: 0, fontSize: "1rem", fontWeight: 500 }}>No schema graph yet</p>
              <p style={{ margin: 0, fontSize: "0.875rem" }}>Add entities and relation fields to visualise the schema</p>
            </div>
          ) : (
            <svg
              ref={svgRef}
              width={dimensions.width}
              height={dimensions.height}
              style={{ display: "block" }}
              onPointerMove={handleSvgPointerMove}
              onPointerUp={handleSvgPointerUp}
              onPointerLeave={handleSvgPointerUp}
            >
              <defs>
                {/* Soft glow for all nodes */}
                <filter id="sg-glow" x="-60%" y="-60%" width="220%" height="220%">
                  <feGaussianBlur stdDeviation="3.5" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>

                {/* Strong glow for hovered / selected */}
                <filter id="sg-glow-strong" x="-100%" y="-100%" width="300%" height="300%">
                  <feGaussianBlur stdDeviation="7" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>

                {/* Arrow markers */}
                <marker id="sg-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
                  <path d="M0,1 L0,7 L7,4 z" fill="rgba(255,255,255,0.7)" />
                </marker>
                <marker id="sg-arrow-missing" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
                  <path d="M0,1 L0,7 L7,4 z" fill="rgba(245,158,11,0.8)" />
                </marker>
              </defs>

              {/* ── Links ───────────────────────────────────────────────── */}
              <g>
                {links.map((link, i) => {
                  const src = nodes.find((n) => n.id === link.source);
                  const tgt = nodes.find((n) => n.id === link.target);
                  if (!src || !tgt) return null;

                  const dx = tgt.x - src.x;
                  const dy = tgt.y - src.y;
                  const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                  const arrowOffset = NODE_R + 6;
                  const x1 = src.x + (dx / dist) * NODE_R;
                  const y1 = src.y + (dy / dist) * NODE_R;
                  const x2 = tgt.x - (dx / dist) * arrowOffset;
                  const y2 = tgt.y - (dy / dist) * arrowOffset;

                  const isAdj = adjacentIds
                    ? adjacentIds.has(link.source) && adjacentIds.has(link.target)
                    : false;

                  // Highlight adjacent links; dim others; default subtle
                  const lineOpacity = hoveredNodeId
                    ? isAdj ? 1 : 0.12
                    : 0.35;
                  const lineStroke = isAdj
                    ? link.isMissing ? "rgba(245,158,11,0.95)" : "rgba(255,255,255,0.95)"
                    : link.isMissing ? "rgba(245,158,11,0.25)" : "rgba(160,160,160,0.25)";
                  const lineWidth = isAdj ? 1.8 : 1.2;

                  const midX = (src.x + tgt.x) / 2;
                  const midY = (src.y + tgt.y) / 2 - 5;

                  return (
                    <g key={i}>
                      <line
                        x1={x1} y1={y1} x2={x2} y2={y2}
                        stroke={lineStroke}
                        strokeWidth={lineWidth}
                        strokeLinecap="round"
                        strokeDasharray={link.isMissing ? "4 3" : undefined}
                        markerEnd={isAdj ? (link.isMissing ? "url(#sg-arrow-missing)" : "url(#sg-arrow)") : undefined}
                        style={{ transition: "stroke 0.2s, opacity 0.2s", opacity: lineOpacity }}
                      />
                      {/* Edge label — only when adjacent on hover */}
                      {isAdj && (
                        <text
                          x={midX}
                          y={midY}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize="9"
                          fontFamily="Inter, system-ui, sans-serif"
                          fill="#F4F4F4"
                          paintOrder="stroke"
                          stroke="rgba(15,10,35,0.85)"
                          strokeWidth="3"
                          style={{ pointerEvents: "none", userSelect: "none" }}
                        >
                          {link.label}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>

              {/* ── Nodes ───────────────────────────────────────────────── */}
              <g>
                {nodes.map((node) => {
                  const isSelected = node.entityId !== null && node.entityId === selectedEntityId;
                  const isHovered = node.id === hoveredNodeId;
                  const isAdj = adjacentIds ? adjacentIds.has(node.id) : false;
                  const dimmed = hoveredNodeId !== null && !isAdj;

                  const color = node.type === "missing" ? MISSING_COLOR : ENTITY_COLOR;
                  const r = isHovered || (isAdj && hoveredNodeId !== null) ? NODE_R + 2 : NODE_R;
                  const nodeOpacity = dimmed ? 0.2 : 1;
                  const glowFilter = isHovered || isSelected ? "url(#sg-glow-strong)" : "url(#sg-glow)";

                  return (
                    <g
                      key={node.id}
                      transform={`translate(${node.x},${node.y})`}
                      style={{ cursor: "grab", opacity: nodeOpacity, transition: "opacity 0.2s" }}
                      onPointerDown={(e) => handleNodePointerDown(node.id, e)}
                      onPointerEnter={() => { if (!dragNodeIdRef.current) setHoveredNodeId(node.id); }}
                      onPointerLeave={() => setHoveredNodeId(null)}
                      onClick={() => {
                        if (!didDragRef.current && node.entityId) onEntitySelect(node.entityId);
                        didDragRef.current = false;
                      }}
                    >
                      {/* Selected ring */}
                      {isSelected && (
                        <circle
                          r={NODE_R + 6}
                          fill="none"
                          stroke={SELECTED_COLOR}
                          strokeWidth="1.5"
                          opacity={0.75}
                          filter="url(#sg-glow-strong)"
                        />
                      )}

                      {/* Node body */}
                      <circle
                        r={r}
                        fill={color}
                        stroke="white"
                        strokeWidth="0.5"
                        filter={glowFilter}
                      />

                      {/* Label */}
                      <text
                        y={NODE_R + 12}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize={isHovered || isAdj ? "11" : "10"}
                        fontWeight="bold"
                        fontFamily="Inter, system-ui, sans-serif"
                        fill={node.type === "missing" ? MISSING_COLOR : "#F4F4F4"}
                        paintOrder="stroke"
                        stroke="rgba(0,0,0,0.85)"
                        strokeWidth="2.5"
                        style={{ userSelect: "none", pointerEvents: "none" }}
                      >
                        {node.label}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          )}

          {/* Legend */}
          {!isEmpty && (
            <Flex
              className="absolute bottom-[0.75rem] left-[0.75rem]"
              gap="0.625rem"
              align="center"
              style={{ pointerEvents: "none" }}
            >
              {[
                { color: ENTITY_COLOR, label: "Entity" },
                { color: MISSING_COLOR, label: "Missing" },
                { color: SELECTED_COLOR, label: "Selected" },
              ].map(({ color, label }) => (
                <Flex key={label} align="center" gap="0.25rem">
                  <div style={{ width: 7, height: 7, borderRadius: "50%", backgroundColor: color, flexShrink: 0 }} />
                  <span style={{ color: "rgba(219,216,216,0.55)", fontSize: "0.65rem" }}>{label}</span>
                </Flex>
              ))}
            </Flex>
          )}

          {/* Hover hint */}
          {!isEmpty && !hoveredNodeId && (
            <span
              style={{
                position: "absolute",
                bottom: "0.75rem",
                right: "0.75rem",
                fontSize: "0.65rem",
                color: "rgba(219,216,216,0.3)",
                pointerEvents: "none",
                userSelect: "none",
              }}
            >
              Hover a node to see connections
            </span>
          )}
        </div>

        {/* Drag-resize handle */}
        {!isExpanded && (
          <div
            onMouseDown={(e) => handleResizeStart(e.clientY)}
            onDoubleClick={() => setManualHeight(null)}
            style={{ height: 6, width: "100%", cursor: "row-resize", display: "flex", alignItems: "center", justifyContent: "center", marginTop: 4, borderRadius: 3, background: "#f3f4f6", transition: "background 0.15s" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#e5e7eb")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "#f3f4f6")}
          >
            <div style={{ width: 32, height: 3, borderRadius: 2, background: "#d1d5db" }} />
          </div>
        )}
      </Stack>
    </>
  );
}

"use client";

import React, { useRef, useEffect, useState } from "react";

interface Node {
  id: string;
  label: string;
  type: string;
  properties: any;
  heat: number;
  dna: any;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface Edge {
  source: string;
  target: string;
  label: string;
  properties: any;
}

interface GalaxyVisualizerProps {
  nodes: Node[];
  edges: Edge[];
  selectedNodeId: string | null;
  onSelectNode: (node: Node | null) => void;
  heatmapEnabled: boolean;
  activePathNodes?: string[]; // Node IDs involved in recent recall reasoning
  isRepairing?: boolean;
  repairTargets?: any[];
}

export default function GalaxyVisualizer({
  nodes,
  edges,
  selectedNodeId,
  onSelectNode,
  heatmapEnabled,
  activePathNodes = [],
  isRepairing = false,
  repairTargets = [],
}: GalaxyVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Graph state in-memory references for simulation loop
  const localNodesRef = useRef<Node[]>([]);
  const localEdgesRef = useRef<Edge[]>([]);

  // Viewport transforms
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [draggedNode, setDraggedNode] = useState<string | null>(null);
  const dragStart = useRef({ x: 0, y: 0 });
  const mousePos = useRef({ x: 0, y: 0 });

  // Particles flowing along edges to simulate thought traffic
  const particlesRef = useRef<Array<{
    edgeIndex: number;
    progress: number; // 0.0 to 1.0
    speed: number;
    size: number;
  }>>([]);

  // Sync props to local references, initializing coordinates if new
  useEffect(() => {
    const nextNodes = nodes.map((n) => {
      const existing = localNodesRef.current.find((ln) => ln.id === n.id);
      return {
        ...n,
        x: existing?.x ?? (Math.random() - 0.5) * 300,
        y: existing?.y ?? (Math.random() - 0.5) * 300,
        vx: existing?.vx ?? 0,
        vy: existing?.vy ?? 0,
      };
    });

    localNodesRef.current = nextNodes;
    localEdgesRef.current = edges;

    // Reset/seed particles
    if (edges.length > 0 && particlesRef.current.length < edges.length * 0.5) {
      const newParticles = [];
      for (let i = 0; i < Math.min(25, edges.length * 2); i++) {
        newParticles.push({
          edgeIndex: Math.floor(Math.random() * edges.length),
          progress: Math.random(),
          speed: 0.005 + Math.random() * 0.01,
          size: 1.5 + Math.random() * 2,
        });
      }
      particlesRef.current = newParticles;
    }
  }, [nodes, edges]);

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Main physics + render loop
  useEffect(() => {
    let animationId: number;

    const tick = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        animationId = requestAnimationFrame(tick);
        return;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;

      // 1. PHYSICS UPDATE
      const ns = localNodesRef.current;
      const es = localEdgesRef.current;

      // Apply forces
      const kRepel = 400; // Coulomb repulsion
      const kAttract = 0.03; // Hooke spring constant
      const linkLength = 80;
      const gravity = 0.02;

      // Coulomb Repulsion (all node pairs)
      for (let i = 0; i < ns.length; i++) {
        const n1 = ns[i];
        for (let j = i + 1; j < ns.length; j++) {
          const n2 = ns[j];
          const dx = n2.x! - n1.x!;
          const dy = n2.y! - n1.y!;
          const distSq = dx * dx + dy * dy + 0.1;
          const dist = Math.sqrt(distSq);

          if (dist < 300) {
            const force = kRepel / distSq;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;

            if (n1.id !== draggedNode) {
              n1.vx! -= fx;
              n1.vy! -= fy;
            }
            if (n2.id !== draggedNode) {
              n2.vx! += fx;
              n2.vy! += fy;
            }
          }
        }
      }

      // Hooke Attraction (along edges)
      for (const e of es) {
        const sourceNode = ns.find((n) => n.id === e.source);
        const targetNode = ns.find((n) => n.id === e.target);

        if (sourceNode && targetNode) {
          const dx = targetNode.x! - sourceNode.x!;
          const dy = targetNode.y! - sourceNode.y!;
          const dist = Math.sqrt(dx * dx + dy * dy) + 0.1;
          const force = kAttract * (dist - linkLength);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          if (sourceNode.id !== draggedNode) {
            sourceNode.vx! += fx;
            sourceNode.vy! += fy;
          }
          if (targetNode.id !== draggedNode) {
            targetNode.vx! -= fx;
            targetNode.vy! -= fy;
          }
        }
      }

      // Gravity towards center and position update
      for (const n of ns) {
        if (n.id === draggedNode) continue;

        // Center gravity
        n.vx! -= n.x! * gravity;
        n.vy! -= n.y! * gravity;

        // Apply friction/damping
        n.vx! *= 0.85;
        n.vy! *= 0.85;

        // Update coordinate
        n.x! += n.vx!;
        n.y! += n.vy!;
      }

      // Repair Animation: Force duplicate pairs to merge
      if (isRepairing && repairTargets && repairTargets.length > 0) {
        for (const target of repairTargets) {
          const n1 = ns.find(n => n.id === target.node1?.id);
          const n2 = ns.find(n => n.id === target.node2?.id);
          if (n1 && n2) {
            const dx = n2.x! - n1.x!;
            const dy = n2.y! - n1.y!;
            const dist = Math.sqrt(dx * dx + dy * dy) + 0.1;
            
            // Strong attraction to pull them together
            const pullForce = 8.0;
            n1.x! += (dx / dist) * pullForce;
            n1.y! += (dy / dist) * pullForce;
            n2.x! -= (dx / dist) * pullForce;
            n2.y! -= (dy / dist) * pullForce;
          }
        }
      }

      // Update active particles
      const particles = particlesRef.current;
      for (const p of particles) {
        if (es.length === 0) break;
        p.progress += p.speed;
        if (p.progress >= 1.0) {
          p.progress = 0;
          p.edgeIndex = Math.floor(Math.random() * es.length);
        }
      }

      // 2. RENDER GRAPH
      ctx.clearRect(0, 0, width, height);

      // Deep space grid background
      ctx.strokeStyle = "rgba(40, 40, 80, 0.15)";
      ctx.lineWidth = 1;
      const gridSize = 50 * scale;
      const startX = (offset.x % gridSize);
      const startY = (offset.y % gridSize);

      for (let x = startX; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = startY; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Save state and apply camera transformations
      ctx.save();
      ctx.translate(width / 2 + offset.x, height / 2 + offset.y);
      ctx.scale(scale, scale);

      // Draw Edges
      for (const e of es) {
        const src = ns.find((n) => n.id === e.source);
        const tgt = ns.find((n) => n.id === e.target);
        if (!src || !tgt) continue;

        const isActive = activePathNodes.includes(e.source) && activePathNodes.includes(e.target);

        ctx.beginPath();
        ctx.moveTo(src.x!, src.y!);
        ctx.lineTo(tgt.x!, tgt.y!);
        ctx.strokeStyle = isActive 
          ? "rgba(239, 68, 68, 0.8)"  // Red glowing for active reasoning paths
          : "rgba(100, 116, 139, 0.25)";
        ctx.lineWidth = isActive ? 2.5 : 1.2;
        ctx.stroke();

        // Edge label (relationship name) on hover or zoom
        if (scale > 0.8) {
          const midX = (src.x! + tgt.x!) / 2;
          const midY = (src.y! + tgt.y!) / 2;
          ctx.font = "8px Outfit, sans-serif";
          ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
          ctx.textAlign = "center";
          ctx.fillText(e.label, midX, midY - 4);
        }
      }

      // Draw active particles running along edges
      for (const p of particles) {
        if (es.length === 0 || p.edgeIndex >= es.length) continue;
        const e = es[p.edgeIndex];
        const src = ns.find((n) => n.id === e.source);
        const tgt = ns.find((n) => n.id === e.target);
        if (!src || !tgt) continue;

        const px = src.x! + (tgt.x! - src.x!) * p.progress;
        const py = src.y! + (tgt.y! - src.y!) * p.progress;

        ctx.beginPath();
        ctx.arc(px, py, p.size, 0, Math.PI * 2);
        
        const isPathPart = activePathNodes.includes(e.source) && activePathNodes.includes(e.target);
        ctx.fillStyle = isPathPart ? "#EF4444" : "#60A5FA"; // Red or blue particle

        ctx.shadowBlur = 10;
        ctx.shadowColor = isPathPart ? "#EF4444" : "#3B82F6";
        ctx.fill();
        ctx.shadowBlur = 0; // reset
      }

      // Draw Nodes
      for (const n of ns) {
        const isSelected = selectedNodeId === n.id;
        const inActivePath = activePathNodes.includes(n.id);
        const radius = isSelected ? 12 : 9;

        // Choose node color
        let color = "#3B82F6"; // default blue
        if (heatmapEnabled) {
          // Heatmap view: Cold (blue) -> Warm (green) -> Hot (orange) -> Critical (red)
          if (n.heat <= 0.1) color = "#1E293B"; // Dark/Cold
          else if (n.heat <= 0.4) color = "#10B981"; // Warm Green
          else if (n.heat <= 0.8) color = "#F59E0B"; // Hot Orange
          else color = "#EF4444"; // Critical Red
        } else {
          // Standard type view
          const t = n.type.toLowerCase();
          if (t === "document" || t === "file") color = "#06B6D4"; // Cyan
          else if (t === "concept") color = "#8B5CF6"; // Purple
          else if (t === "chunk") color = "#EAB308"; // Yellow
          else if (t === "person") color = "#EF4444"; // Red
          else if (t === "organization") color = "#3B82F6"; // Blue
          else color = "#10B981"; // Green for other entities
        }

        // Is this node part of a repair merge?
        let isRepairTarget = false;
        if (isRepairing && repairTargets) {
          isRepairTarget = repairTargets.some(t => t.node1?.id === n.id || t.node2?.id === n.id);
        }

        // Draw node glow
        ctx.beginPath();
        ctx.arc(n.x!, n.y!, radius, 0, Math.PI * 2);
        
        if (isRepairTarget) {
          // Pulsing green merge glow
          const pulse = Math.abs(Math.sin(Date.now() / 150));
          ctx.shadowBlur = 20 + pulse * 15;
          ctx.shadowColor = "#10B981";
          ctx.fillStyle = "#10B981"; // Force green during repair
          ctx.globalAlpha = Math.max(0.2, 1 - pulse * 0.5); // Fade effect
        } else {
          ctx.shadowBlur = isSelected ? 20 : (inActivePath ? 15 : 6);
          ctx.shadowColor = inActivePath ? "#EF4444" : color;
          ctx.fillStyle = color;
          ctx.globalAlpha = 1.0;
        }
        
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1.0;

        // Outer border for selection
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(n.x!, n.y!, radius + 3, 0, Math.PI * 2);
          ctx.strokeStyle = "#FFFFFF";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        } else if (inActivePath) {
          ctx.beginPath();
          ctx.arc(n.x!, n.y!, radius + 3, 0, Math.PI * 2);
          ctx.strokeStyle = "#EF4444";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Node Label
        ctx.font = isSelected ? "bold 10px Outfit, sans-serif" : "9px Outfit, sans-serif";
        ctx.fillStyle = isSelected ? "#FFFFFF" : "rgba(248, 250, 252, 0.85)";
        ctx.textAlign = "center";
        ctx.fillText(n.label, n.x!, n.y! + radius + 13);
      }

      ctx.restore();
      animationId = requestAnimationFrame(tick);
    };

    animationId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationId);
  }, [selectedNodeId, scale, offset, draggedNode, heatmapEnabled, activePathNodes, isRepairing, repairTargets]);

  // Handle Mouse Events (Zooming)
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = 1.1;
    const newScale = e.deltaY < 0 ? scale * zoomFactor : scale / zoomFactor;
    setScale(Math.max(0.2, Math.min(3, newScale)));
  };

  // Convert Screen Coordinates to Graph coordinates
  const screenToGraph = (screenX: number, screenY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const x = screenX - rect.left - canvas.width / 2 - offset.x;
    const y = screenY - rect.top - canvas.height / 2 - offset.y;
    return { x: x / scale, y: y / scale };
  };

  // Handle Mouse Click Down
  const handleMouseDown = (e: React.MouseEvent) => {
    const { x, y } = screenToGraph(e.clientX, e.clientY);
    
    // Check if clicked a node
    const ns = localNodesRef.current;
    let clicked: Node | null = null;
    
    for (const n of ns) {
      const dx = n.x! - x;
      const dy = n.y! - y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 15) {
        clicked = n;
        break;
      }
    }

    if (clicked) {
      setDraggedNode(clicked.id);
      onSelectNode(clicked);
    } else {
      setIsDragging(true);
      dragStart.current = { x: e.clientX - offset.x, y: e.clientY - offset.y };
    }
  };

  // Handle Mouse Move
  const handleMouseMove = (e: React.MouseEvent) => {
    mousePos.current = { x: e.clientX, y: e.clientY };

    if (draggedNode) {
      const { x, y } = screenToGraph(e.clientX, e.clientY);
      const ns = localNodesRef.current;
      const node = ns.find((n) => n.id === draggedNode);
      if (node) {
        node.x = x;
        node.y = y;
        node.vx = 0;
        node.vy = 0;
      }
    } else if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.current.x,
        y: e.clientY - dragStart.current.y,
      });
    }
  };

  // Handle Mouse Up
  const handleMouseUp = () => {
    setDraggedNode(null);
    setIsDragging(false);
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full min-h-[480px] bg-slate-950 overflow-hidden cursor-grab active:cursor-grabbing border border-slate-800 rounded-2xl shadow-inner shadow-black/80"
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* HUD Info */}
      <div className="absolute top-4 left-4 z-10 bg-slate-900/80 backdrop-blur-md border border-slate-700 px-3 py-1.5 rounded-lg text-xs text-slate-300 font-mono pointer-events-none">
        Zoom: {Math.round(scale * 100)}% | Nodes: {nodes.length} | Edges: {edges.length}
      </div>

      <div className="absolute top-4 right-4 z-10 bg-slate-900/80 backdrop-blur-md border border-slate-700 px-3 py-1.5 rounded-lg text-xs text-slate-300 pointer-events-none flex flex-col gap-1 shadow-md">
        <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold border-b border-slate-800 pb-0.5 mb-1">
          {heatmapEnabled ? "Recency Heatmap" : "Legend"}
        </div>
        {heatmapEnabled ? (
          <div className="flex flex-col gap-1 font-mono text-[10px]">
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#EF4444] shadow shadow-[#EF4444]"></span> Critical</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#F59E0B] shadow shadow-[#F59E0B]"></span> Hot</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#10B981] shadow shadow-[#10B981]"></span> Warm</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#1E293B]"></span> Cold</div>
          </div>
        ) : (
          <div className="flex flex-col gap-1 text-[10px]">
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#06B6D4] shadow shadow-[#06B6D4]"></span> Document</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#8B5CF6] shadow shadow-[#8B5CF6]"></span> Concept</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#EAB308] shadow shadow-[#EAB308]"></span> Chunk</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#EF4444] shadow shadow-[#EF4444]"></span> Person</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#3B82F6] shadow shadow-[#3B82F6]"></span> Organization</div>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  );
}

"use client";

import { useEffect, useRef } from "react";

// Decorative "building your knowledge graph" visual for the preparing
// screen — a living particle network (same canvas/requestAnimationFrame
// technique as TetrisBackground): nodes spawn in one at a time up to a
// target count, drift slowly, bounce softly off the edges, and draw a line
// to any neighbor within range — connections form and dissolve as nodes
// move, so the graph never looks static. Purely decorative: StepPreparing's
// checklist is the real progress signal, this never has to reflect actual
// state.
const WIDTH = 280;
const HEIGHT = 170;
const TARGET_NODE_COUNT = 16;
const SPAWN_INTERVAL_MS = 260;
const LINK_DISTANCE = 85;
const DRIFT_SPEED = 0.012; // px/ms
const COLORS = ["#BC9BFF", "#BC9BFF", "#BC9BFF", "#8CFF86"]; // mostly purple, occasional green highlight

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  color: string;
  bornAt: number;
}

function spawnNode(now: number): Node {
  const angle = Math.random() * Math.PI * 2;
  return {
    x: Math.random() * WIDTH,
    y: Math.random() * HEIGHT,
    vx: Math.cos(angle) * DRIFT_SPEED,
    vy: Math.sin(angle) * DRIFT_SPEED,
    r: 2.5 + Math.random() * 2.5,
    color: COLORS[Math.floor(Math.random() * COLORS.length)],
    bornAt: now,
  };
}

export function GraphBuildingAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    // Fixed logical size, scaled for device pixel ratio so lines stay crisp.
    const dpr = window.devicePixelRatio || 1;
    canvas.width = WIDTH * dpr;
    canvas.height = HEIGHT * dpr;
    ctx.scale(dpr, dpr);

    const nodes: Node[] = [];
    let lastSpawn = 0;
    let raf = 0;
    let lastFrame = 0;

    const tick = (now: number) => {
      const dt = lastFrame ? now - lastFrame : 16;
      lastFrame = now;

      if (nodes.length < TARGET_NODE_COUNT && now - lastSpawn > SPAWN_INTERVAL_MS) {
        nodes.push(spawnNode(now));
        lastSpawn = now;
      }

      ctx.clearRect(0, 0, WIDTH, HEIGHT);

      for (const n of nodes) {
        n.x += n.vx * dt;
        n.y += n.vy * dt;
        if (n.x < n.r || n.x > WIDTH - n.r) n.vx *= -1;
        if (n.y < n.r || n.y > HEIGHT - n.r) n.vy *= -1;
        n.x = Math.min(Math.max(n.x, n.r), WIDTH - n.r);
        n.y = Math.min(Math.max(n.y, n.r), HEIGHT - n.r);
      }

      // Edges — drawn first so nodes sit on top.
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dist = Math.hypot(a.x - b.x, a.y - b.y);
          if (dist > LINK_DISTANCE) continue;
          const alpha = (1 - dist / LINK_DISTANCE) * 0.5;
          ctx.strokeStyle = `rgba(188,155,255,${alpha})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      // Nodes — fade in over their first 400ms so spawning is never a hard pop.
      for (const n of nodes) {
        const fadeIn = Math.min(1, (now - n.bornAt) / 400);
        ctx.globalAlpha = fadeIn;
        ctx.fillStyle = n.color;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      style={{ width: WIDTH, height: HEIGHT, maxWidth: "100%" }}
    />
  );
}

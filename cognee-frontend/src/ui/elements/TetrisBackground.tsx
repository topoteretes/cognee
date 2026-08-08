"use client";

import { useRef, useEffect } from "react";

const CELL = 33;
const TETRIS_COLORS = ["#8CFF86","#72E870","#5FD660","#A8FFB5","#C4F5C1","#F4F4F4","#9CA3A1"];
const SHAPES = [
  [[1,1,1,1]],
  [[1,1],[1,1]],
  [[0,1,0],[1,1,1]],
  [[0,1,1],[1,1,0]],
  [[1,1,0],[0,1,1]],
  [[1,0],[1,0],[1,1]],
  [[0,1],[0,1],[1,1]],
];

type FallingPiece = {
  shape: number[][];
  color: string;
  col: number;
  row: number;
  intervalMs: number;
  lastDropTime: number;
  opacity: number;
};

export function TetrisBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    type LandedCell = { color: string; opacity: number; clearAt: number };
    const landed = new Map<number, LandedCell>();
    const lKey = (r: number, c: number) => r * 500 + c;

    function isLanded(gr: number, gc: number, now: number) {
      const cell = landed.get(lKey(gr, gc));
      return !!cell && cell.clearAt > now;
    }

    function isActive(gr: number, gc: number, pieces: FallingPiece[], skip: number) {
      for (let i = 0; i < pieces.length; i++) {
        if (i === skip) continue;
        const p = pieces[i];
        const lr = gr - p.row;
        if (lr < 0 || lr >= p.shape.length) continue;
        const lc = gc - p.col;
        if (lc < 0 || lc >= p.shape[lr].length) continue;
        if (p.shape[lr][lc]) return true;
      }
      return false;
    }

    function wouldCollide(shape: number[][], row: number, col: number, rows: number, pieces: FallingPiece[], skip: number, now: number) {
      for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
          if (!shape[r][c]) continue;
          const gr = row + r; const gc = col + c;
          if (gr >= rows) return true;
          if (isLanded(gr, gc, now)) return true;
          if (isActive(gr, gc, pieces, skip)) return true;
        }
      }
      return false;
    }

    function doLand(p: FallingPiece, now: number) {
      for (let r = 0; r < p.shape.length; r++) {
        for (let c = 0; c < p.shape[r].length; c++) {
          if (!p.shape[r][c] || p.row + r < 0) continue;
          landed.set(lKey(p.row + r, p.col + c), {
            color: p.color, opacity: p.opacity,
            clearAt: now + 2000 + Math.random() * 1500,
          });
        }
      }
    }

    function spawnPiece(cols: number, now: number, pieces: FallingPiece[], skip: number): FallingPiece {
      const shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];
      const maxCol = Math.max(1, cols - shape[0].length);
      let col = Math.floor(Math.random() * maxCol);
      for (let attempt = 0; attempt < 15; attempt++) {
        const tc = Math.floor(Math.random() * maxCol);
        let free = true;
        outer: for (let r = 0; r < shape.length; r++)
          for (let c = 0; c < shape[r].length; c++) {
            if (!shape[r][c]) continue;
            if (isActive(-shape.length + r, tc + c, pieces, skip)) { free = false; break outer; }
          }
        if (free) { col = tc; break; }
      }
      return {
        shape, col, row: -shape.length,
        color: TETRIS_COLORS[Math.floor(Math.random() * TETRIS_COLORS.length)],
        intervalMs: 250, lastDropTime: now,
        opacity: 0.12 + Math.random() * 0.18,
      };
    }

    const pieces: FallingPiece[] = [];
    let rafId: number;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const getCols = () => Math.floor(canvas.width / CELL);
    const getRows = () => Math.floor(canvas.height / CELL);

    const now = performance.now();
    for (let i = 0; i < 10; i++) {
      const p = spawnPiece(getCols(), now, pieces, i);
      p.row = Math.floor(Math.random() * (getRows() + p.shape.length)) - p.shape.length;
      p.lastDropTime = now - Math.random() * p.intervalMs;
      pieces.push(p);
    }

    const draw = (timestamp: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const cols = getCols();
      const rows = getRows();

      for (const [k, cell] of landed)
        if (cell.clearAt <= timestamp) landed.delete(k);

      for (let i = 0; i < pieces.length; i++) {
        const p = pieces[i];
        if (timestamp - p.lastDropTime >= p.intervalMs) {
          if (wouldCollide(p.shape, p.row + 1, p.col, rows, pieces, i, timestamp)) {
            if (p.row + p.shape.length > 0) doLand(p, timestamp);
            Object.assign(p, spawnPiece(cols, timestamp, pieces, i));
          } else {
            p.row += 1;
            p.lastDropTime = timestamp;
          }
        }
        if (p.row > rows + p.shape.length)
          Object.assign(p, spawnPiece(cols, timestamp, pieces, i));

        ctx.globalAlpha = p.opacity;
        for (let r = 0; r < p.shape.length; r++)
          for (let c = 0; c < p.shape[r].length; c++) {
            if (!p.shape[r][c]) continue;
            ctx.fillStyle = p.color;
            ctx.fillRect((p.col + c) * CELL + 1, (p.row + r) * CELL + 1, CELL - 2, CELL - 2);
          }
      }
      ctx.globalAlpha = 1;

      for (const [k, cell] of landed) {
        ctx.globalAlpha = cell.opacity;
        ctx.fillStyle = cell.color;
        ctx.fillRect((k % 500) * CELL + 1, Math.floor(k / 500) * CELL + 1, CELL - 2, CELL - 2);
      }
      ctx.globalAlpha = 1;

      rafId = requestAnimationFrame(draw);
    };
    rafId = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      style={{ pointerEvents: "none", position: "absolute", inset: 0, zIndex: 1, width: "100%", height: "100%" }}
    />
  );
}

export const darkPageStyle: React.CSSProperties = {
  backgroundColor: "#000000",
  backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
  backgroundSize: "33px 33px",
  position: "relative",
  overflow: "hidden",
};

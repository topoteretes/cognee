"use client";

import { useEffect, useRef } from "react";

// Replicates the falling-tetromino canvas from the website's landing hero
// (cognee-website, EditorialHero BackgroundLayers): 33px cells aligned to the
// page grid, classic shapes, lavender palette, one row drop per 250ms.
const CELL = 33;
const DROP_MS = 250;
const PIECE_COUNT = 18;
const COLORS = ["#BC9BFF", "#A380EA", "#916DD9", "#DDCCFF", "#E9DFFB", "#F4F4F4", "#9CA3A1"];
const SHAPES: [number, number][][] = [
  [[0, 0], [1, 0], [2, 0], [3, 0]], // I
  [[0, 0], [1, 0], [0, 1], [1, 1]], // O
  [[0, 0], [1, 0], [2, 0], [1, 1]], // T
  [[0, 0], [0, 1], [1, 1], [2, 1]], // J
  [[2, 0], [0, 1], [1, 1], [2, 1]], // L
  [[1, 0], [2, 0], [0, 1], [1, 1]], // S
  [[0, 0], [1, 0], [1, 1], [2, 1]], // Z
];

interface Piece {
  shape: [number, number][];
  col: number;
  row: number;
  color: string;
  opacity: number;
  lastDrop: number;
}

export default function TetrisBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    let width = 0;
    let height = 0;
    const resize = () => {
      width = canvas.width = canvas.offsetWidth;
      height = canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const spawn = (now: number): Piece => ({
      shape: SHAPES[Math.floor(Math.random() * SHAPES.length)],
      col: Math.floor(Math.random() * Math.max(1, Math.floor(width / CELL) - 4)),
      row: -4 - Math.floor(Math.random() * 20),
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      opacity: 0.12 + Math.random() * 0.18,
      lastDrop: now,
    });

    let pieces: Piece[] = [];
    let raf = 0;
    const tick = (now: number) => {
      if (pieces.length === 0) pieces = Array.from({ length: PIECE_COUNT }, () => spawn(now));
      ctx.clearRect(0, 0, width, height);
      pieces = pieces.map((p) => {
        if (now - p.lastDrop >= DROP_MS) {
          p.row += 1;
          p.lastDrop = now;
        }
        return p.row * CELL > height ? spawn(now) : p;
      });
      for (const p of pieces) {
        ctx.globalAlpha = p.opacity;
        ctx.fillStyle = p.color;
        for (const [dx, dy] of p.shape) {
          // 1px inset so the page grid shows through between cells
          ctx.fillRect((p.col + dx) * CELL + 1, (p.row + dy) * CELL + 1, CELL - 2, CELL - 2);
        }
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 z-[1] h-full w-full"
    />
  );
}

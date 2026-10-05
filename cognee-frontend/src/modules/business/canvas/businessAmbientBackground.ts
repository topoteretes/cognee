// A faint, slowly drifting starfield behind the graph — new in this port
// (no source equivalent), giving the canvas some depth/atmosphere at rest
// instead of a flat gradient. Drawn in screen space, before the camera
// transform, the same layer as the filaments.

// mulberry32: a tiny deterministic PRNG, seeded once at module load — a
// fixed field that survives HMR and re-renders, rather than Math.random()
// reshuffling every star's position on every dev reload.
function mulberry32(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface AmbientStar {
  x: number;
  y: number;
  r: number;
  phase: number;
  driftSpeed: number;
}

const STAR_COUNT = 90;
const SEED = 42;
const DRIFT_PERIOD_MS = 60000;
const TWINKLE_PERIOD_MS = 3200;

function generateAmbientStars(): AmbientStar[] {
  const rand = mulberry32(SEED);
  return Array.from({ length: STAR_COUNT }, () => ({
    x: rand(),
    y: rand(),
    r: 0.5 + rand() * 1.3,
    phase: rand() * Math.PI * 2,
    driftSpeed: 0.4 + rand() * 0.8,
  }));
}

const AMBIENT_STARS: AmbientStar[] = generateAmbientStars();

export function drawAmbientBackground(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  now: number,
  reducedMotion: boolean,
): void {
  // Reduced motion keeps the field itself (it's atmosphere, not decoration
  // to strip) but freezes drift/twinkle rather than hiding it outright.
  const motionNow = reducedMotion ? 0 : now;
  AMBIENT_STARS.forEach((s) => {
    const driftX = (((s.x + (motionNow / DRIFT_PERIOD_MS) * s.driftSpeed) % 1) + 1) % 1;
    const x = driftX * width;
    const y = s.y * height;
    const twinkle = 0.5 + 0.5 * Math.sin(motionNow / TWINKLE_PERIOD_MS + s.phase);
    ctx.beginPath();
    ctx.arc(x, y, s.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(126,140,166,${0.1 + 0.12 * twinkle})`;
    ctx.fill();
  });
}

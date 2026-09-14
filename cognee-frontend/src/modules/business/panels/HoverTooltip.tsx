"use client";

import type { SceneHit } from "../canvas/businessHitTest";
import { setsOf } from "../computeBrainState";

interface HoverTooltipProps {
  hit: SceneHit;
  docCount: number;
  // How many semantic links touch this entity — a quick "how important is
  // this to the model" read, new in this port alongside docCount.
  connectionCount: number;
  x: number;
  y: number;
  // Same coordinate space as x/y (the positioning container, not the
  // viewport) — clamping against window.innerWidth here would reintroduce
  // the same offset bug x/y themselves were fixed for.
  containerWidth: number;
}

const MAX_MEMBER_NAMES = 5;

// Ports the mousemove hovercard (customer_tutorial.html:6572-6604) — below
// the zoom threshold this is a type node's member peek ("N records" + a
// name sample + "zoom in to explore"); above it, an entity's name/type/
// source/doc-count. Below k<1.4 every hover IS a type-node hit — a version
// of this that only ever rendered the entity case left the type layer
// entirely mute to hover, which is what "tooltip doesn't work" was actually
// reporting: the default zoom level's hover never showed anything.
export default function HoverTooltip({ hit, docCount, connectionCount, x, y, containerWidth }: HoverTooltipProps) {
  if (!hit) return null;
  const maxLeft = containerWidth > 0 ? containerWidth - 260 : x + 14;
  const style = { left: Math.min(x + 14, maxLeft), top: y + 14 };
  const className = "absolute z-10 max-w-[240px] rounded-[10px] border border-[#2A3652] bg-[#1A2438] px-2.5 py-2 text-[11px] pointer-events-none";

  if (hit.kind === "type") {
    const tn = hit.node;
    const names = tn.members.slice(0, MAX_MEMBER_NAMES).map((m) => String(m.name || "")).join(", ");
    return (
      <div className={className} style={style}>
        <b className="text-[#E9EEF6]">{tn.name}</b> · {tn.members.length} record{tn.members.length === 1 ? "" : "s"}
        <br />
        <span className="text-[#7E8CA6]">
          {names}{tn.members.length > MAX_MEMBER_NAMES ? "…" : ""}
          <br />
          zoom in to explore
        </span>
      </div>
    );
  }

  const entity = hit.node;
  const sets = setsOf(entity);
  return (
    <div className={className} style={style}>
      <b className="text-[#E9EEF6]">{entity.name}</b>
      <br />
      <span className="text-[#7E8CA6]">
        {entity.type || ""}
        {sets.length ? ` · from ${sets.join(", ")}` : ""}
        {docCount ? ` · seen in ${docCount} places` : ""}
        {connectionCount ? ` · ${connectionCount} connection${connectionCount === 1 ? "" : "s"}` : ""}
      </span>
    </div>
  );
}

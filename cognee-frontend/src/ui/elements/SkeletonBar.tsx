export default function SkeletonBar({
  width,
  height = 10,
}: {
  width: number | string;
  height?: number | string;
}) {
  // One element, no inline <style>. The keyframes live in globals.css: this
  // component is instantiated 68 times by a single skeleton, and each instance
  // was emitting its own identical @keyframes block.
  return (
    <span
      style={{
        display: "inline-block",
        width,
        height,
        borderRadius: 4,
        background: "rgba(237,236,234,0.18)",
        animation: "skpulse 1.4s ease-in-out infinite",
      }}
    />
  );
}

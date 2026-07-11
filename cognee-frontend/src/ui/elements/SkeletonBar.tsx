export default function SkeletonBar({
  width,
  height = 10,
}: {
  width: number | string;
  height?: number | string;
}) {
  return (
    <>
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
      <style>{`@keyframes skpulse { 0%,100% { opacity: 0.35; } 50% { opacity: 0.7; } }`}</style>
    </>
  );
}

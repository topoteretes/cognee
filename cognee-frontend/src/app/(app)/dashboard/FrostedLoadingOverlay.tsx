"use client";


export default function FrostedLoadingOverlay({
  visible,
  title,
  subtitle,
}: {
  visible: boolean;
  title?: string;
  subtitle?: string;
}) {
  if (!visible) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "2.5rem",
        background: "#FFFFFF",
      }}
    >
      <video
        src="/videos/mascot-waiting.mp4"
        autoPlay
        loop
        muted
        playsInline
        style={{ width: 200, height: "auto" }}
      />

    </div>
  );
}

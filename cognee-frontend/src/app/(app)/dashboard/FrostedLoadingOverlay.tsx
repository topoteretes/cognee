"use client";

import { tokens } from "@/ui/theme/tokens";

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
        background:
          "linear-gradient(135deg, rgba(255,255,255,0.6) 0%, rgba(240,235,255,0.5) 50%, rgba(255,255,255,0.6) 100%)",
        backdropFilter: "blur(40px) saturate(200%)",
        WebkitBackdropFilter: "blur(40px) saturate(200%)",
      }}
    >
      <div
        style={{
          position: "relative",
          width: 80,
          height: 80,
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            border: "3px solid rgba(101, 16, 244, 0.1)",
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            border: "3px solid transparent",
            borderTopColor: tokens.purple,
            animation: "spin 1s cubic-bezier(0.4, 0, 0.2, 1) infinite",
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: 12,
            borderRadius: "50%",
            border: "3px solid transparent",
            borderTopColor: tokens.green,
            animation: "spin 1.4s cubic-bezier(0.4, 0, 0.2, 1) infinite reverse",
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: 24,
            borderRadius: "50%",
            border: "3px solid transparent",
            borderTopColor: tokens.purple,
            opacity: 0.5,
            animation: "spin 1.8s cubic-bezier(0.4, 0, 0.2, 1) infinite",
          }}
        />
      </div>

      <div style={{ textAlign: "center" }}>
        <p
          style={{
            margin: 0,
            fontSize: "1.125rem",
            fontWeight: 500,
            color: tokens.textDark,
            letterSpacing: "0.01em",
            animation: "fadeText 2.5s ease-in-out infinite",
          }}
        >
          {title || "Preparing your workspace"}
        </p>
        <p
          style={{
            margin: "0.5rem 0 0",
            fontSize: "0.875rem",
            fontWeight: 400,
            color: tokens.textSecondary,
            animation: "fadeText 2.5s ease-in-out infinite 0.3s",
          }}
        >
          {subtitle || "Loading your datasets and knowledge graphs..."}
        </p>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes fadeText {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.35; }
        }
      `}</style>
    </div>
  );
}

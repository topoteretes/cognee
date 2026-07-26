import { PropsWithChildren } from "react";

// Translucent glass card that lifts the auth forms above the animated hero
// background — dark enough to anchor the content, transparent enough that the
// grid and falling pieces stay visible behind it. Radius matches the website
// card token (16px).
export default function AuthCard({ children }: PropsWithChildren) {
  return (
    <div
      className="flex w-full max-w-[28rem] flex-col items-center gap-[1.5rem] rounded-2xl px-8 py-10"
      style={{
        background: "rgba(0,0,0,0.55)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        border: "1px solid rgba(255,255,255,0.12)",
        boxShadow: "0 24px 80px rgba(0,0,0,0.55)",
      }}
    >
      {children}
    </div>
  );
}

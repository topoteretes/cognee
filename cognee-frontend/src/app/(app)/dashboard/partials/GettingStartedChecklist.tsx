"use client";

interface ChecklistItem {
  label: string;
  done: boolean;
}

function ChecklistStep({ label, done, isNext, stepNumber }: ChecklistItem & { isNext: boolean; stepNumber: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 0" }}>
      {done ? (
        <svg width="18" height="18" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
          <circle cx="8" cy="8" r="8" fill="#22C55E" />
          <path d="M4.5 8.5L6.8 10.8L11.5 5.5" stroke="#0A0A0A" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ) : (
        <span
          style={{
            width: 18,
            height: 18,
            borderRadius: "50%",
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 10,
            fontWeight: 700,
            border: `1.5px solid ${isNext ? "#BC9BFF" : "rgba(237,236,234,0.2)"}`,
            color: isNext ? "#BC9BFF" : "rgba(237,236,234,0.35)",
          }}
        >
          {stepNumber}
        </span>
      )}
      <span
        style={{
          fontSize: 14,
          fontWeight: isNext ? 600 : 400,
          color: done ? "rgba(237,236,234,0.4)" : isNext ? "#EDECEA" : "rgba(237,236,234,0.4)",
          textDecoration: done ? "line-through" : "none",
        }}
      >
        {label}
      </span>
    </div>
  );
}

// Distinct from AgentConnectionSection's cards (which teach *how* to connect
// an agent) — this tracks *whether* the user has actually done the three
// things that predict retention, and disappears once they have. Activity is
// judged the same way useOnboardingRedirect does (pipeline runs / sessions),
// never by dataset count/name — the onboarding-uploaded dataset and the
// auto-provisioned empty placeholder share the same "default_dataset" name,
// so dataset presence alone can't tell them apart.
export function GettingStartedChecklist({ items }: { items: ChecklistItem[] }) {
  if (items.every((item) => item.done)) return null;

  const doneCount = items.filter((item) => item.done).length;
  const nextIndex = items.findIndex((item) => !item.done);
  const progressPct = Math.round((doneCount / items.length) * 100);

  return (
    <div style={{ background: "rgba(188,155,255,0.06)", border: "1px solid rgba(188,155,255,0.25)", borderRadius: 12, padding: "18px 20px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <span style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA" }}>Set up your workspace</span>
        <span style={{ fontSize: 12, fontWeight: 600, color: "#BC9BFF", fontVariantNumeric: "tabular-nums" }}>
          {doneCount} / {items.length}
        </span>
      </div>
      <div style={{ height: 5, borderRadius: 3, background: "rgba(255,255,255,0.08)", overflow: "hidden", marginBottom: 14 }}>
        <div style={{ height: "100%", width: `${progressPct}%`, background: "#BC9BFF", borderRadius: 3, transition: "width 300ms ease" }} />
      </div>
      <div>
        {items.map((item, i) => (
          <ChecklistStep key={item.label} label={item.label} done={item.done} isNext={i === nextIndex} stepNumber={i + 1} />
        ))}
      </div>
    </div>
  );
}

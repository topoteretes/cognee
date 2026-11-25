"use client";

interface NavbarSectionLabelProps {
  label: string;
  railed: boolean;
}

// In the collapsed rail there is no room for the word, so the section reads as
// a divider instead — the grouping survives, the label does not.
export default function NavbarSectionLabel({ label, railed }: NavbarSectionLabelProps): React.JSX.Element {
  if (railed) {
    return <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "0 8px 8px" }} />;
  }

  return (
    <div
      className="px-3 mb-1"
      style={{
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: 0.5,
        color: "rgba(255,255,255,0.3)",
        textTransform: "uppercase",
      }}
    >
      {label}
    </div>
  );
}

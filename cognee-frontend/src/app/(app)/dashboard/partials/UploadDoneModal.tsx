"use client";

import Modal from "@/ui/elements/Modal/Modal";

interface UploadDoneModalProps {
  datasetName: string;
  datasetId: string;
  onClose: () => void;
  onNavigate: (path: string) => void;
}

interface ActionRow {
  label: string;
  sublabel: string;
  path: string;
  icon: React.ReactElement;
}

const ACTION_ROWS: ActionRow[] = [
  {
    label: "Search your data",
    sublabel: "Ask questions about your knowledge graph",
    path: "/search",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-cognee-lavender-tint-60)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
    ),
  },
  {
    label: "Explore the knowledge graph",
    sublabel: "Open the full graph visualization",
    path: "/knowledge-graph",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-cognee-lavender-tint-60)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M9 3v18" />
      </svg>
    ),
  },
];

export function UploadDoneModal({
  datasetName,
  datasetId,
  onClose,
  onNavigate,
}: UploadDoneModalProps): React.ReactElement {
  return (
    <Modal isOpen onClose={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-done-title"
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "rgba(15,15,15,0.92)",
          backdropFilter: "blur(16px)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 12,
          padding: 28,
          width: 440,
          maxWidth: "calc(100vw - 32px)",
          display: "flex",
          flexDirection: "column",
          gap: 20,
          boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
        }}
      >
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 36, height: 36, borderRadius: 8, background: "rgba(34,197,94,0.15)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22C55E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 6L9 17l-5-5" />
            </svg>
          </div>
          <div>
            <h2 id="upload-done-title" style={{ fontSize: 17, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Knowledge graph built</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.65)", margin: 0 }}>&ldquo;{datasetName}&rdquo; is now searchable.</p>
          </div>
        </div>

        {/* Action rows */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {ACTION_ROWS.map((row) => (
            <button
              key={row.path}
              onClick={() => onNavigate(row.path)}
              className="cursor-pointer hover:bg-white/10"
              style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "12px 14px",
                borderRadius: 8,
                border: "1px solid rgba(255,255,255,0.1)",
                background: "rgba(255,255,255,0.06)",
                textAlign: "left",
                fontFamily: "inherit",
                width: "100%",
              }}
            >
              {row.icon}
              <div>
                <div style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>{row.label}</div>
                <div style={{ fontSize: 12, color: "rgba(237,236,234,0.65)" }}>{row.sublabel}</div>
              </div>
            </button>
          ))}

          {/* Dataset-specific inspect row (needs dynamic path) */}
          <button
            onClick={() => onNavigate(`/datasets/${datasetId}`)}
            className="cursor-pointer hover:bg-white/10"
            style={{
              display: "flex", alignItems: "center", gap: 10,
              padding: "12px 14px",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.1)",
              background: "rgba(255,255,255,0.06)",
              textAlign: "left",
              fontFamily: "inherit",
              width: "100%",
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-cognee-lavender-tint-60)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="6" cy="6" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="12" cy="18" r="3" />
              <line x1="8.5" y1="7.5" x2="10.5" y2="16" /><line x1="15.5" y1="7.5" x2="13.5" y2="16" />
            </svg>
            <div>
              <div style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Inspect the knowledge graph</div>
              <div style={{ fontSize: 12, color: "rgba(237,236,234,0.65)" }}>View entities and relationships</div>
            </div>
          </button>
        </div>

        <button
          onClick={onClose}
          className="cursor-pointer hover:bg-white/10"
          style={{
            background: "none",
            border: "1px solid rgba(255,255,255,0.2)",
            borderRadius: 8,
            padding: "8px 16px",
            fontSize: 13,
            fontWeight: 500,
            color: "rgba(237,236,234,0.65)",
            fontFamily: "inherit",
            alignSelf: "flex-end",
          }}
        >
          Stay here
        </button>
      </div>
    </Modal>
  );
}

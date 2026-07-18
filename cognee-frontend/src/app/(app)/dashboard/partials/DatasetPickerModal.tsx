"use client";

import Modal from "@/ui/elements/Modal/Modal";
import type { Dataset } from "@/ui/layout/FilterContext";

interface DatasetPickerModalProps {
  open: boolean;
  datasets: Dataset[];
  pendingFiles: File[];
  onPick: (ds: Dataset) => void;
  onClose: () => void;
}

export function DatasetPickerModal({
  open,
  datasets,
  pendingFiles,
  onPick,
  onClose,
}: DatasetPickerModalProps): React.ReactElement {
  return (
    <Modal isOpen={open} onClose={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-picker-title"
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "rgba(15,15,15,0.92)",
          backdropFilter: "blur(16px)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 12,
          padding: 24,
          width: 420,
          maxWidth: "calc(100vw - 32px)",
          display: "flex",
          flexDirection: "column",
          gap: 16,
          boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
        }}
      >
        <h2 id="dataset-picker-title" style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>
          Upload to which brain?
        </h2>
        <p style={{ fontSize: 13, color: "rgba(237,236,234,0.65)", margin: 0 }}>
          {pendingFiles.length} file{pendingFiles.length !== 1 ? "s" : ""} selected. Choose a brain to upload to.
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 300, overflow: "auto" }}>
          {datasets.map((ds) => (
            <button
              key={ds.id}
              onClick={() => onPick(ds)}
              className="cursor-pointer hover:bg-white/10"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "12px 14px",
                borderRadius: 8,
                border: "1px solid rgba(255,255,255,0.1)",
                background: "none",
                textAlign: "left",
                fontFamily: "inherit",
                width: "100%",
              }}
            >
              <div style={{ width: 8, height: 8, borderRadius: 2, background: "var(--color-cognee-purple)", flexShrink: 0 }} />
              <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>{ds.name}</span>
            </button>
          ))}
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={onClose}
            className="cursor-pointer hover:bg-white/10"
            style={{
              background: "transparent",
              border: "1px solid rgba(255,255,255,0.2)",
              borderRadius: 8,
              padding: "8px 16px",
              fontSize: 13,
              fontWeight: 500,
              color: "rgba(237,236,234,0.65)",
              fontFamily: "inherit",
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    </Modal>
  );
}

"use client";

import type { ReactElement, ReactNode } from "react";
import { Loader } from "@mantine/core";
import ModalShell from "./ModalShell";

// Generic confirm-and-delete modal shared by the "delete document" and
// "delete brain" flows. The message is passed as a node so each caller can
// bold the target name inline.
export default function DeleteConfirmModal({
  title,
  message,
  onConfirm,
  onCancel,
  busy = false,
}: {
  title: string;
  message: ReactNode;
  onConfirm: () => void;
  onCancel: () => void;
  busy?: boolean;
}): ReactElement {
  return (
    <ModalShell onClose={onCancel}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>{title}</h2>
      <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>{message}</p>
      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
        <button onClick={onCancel} className="cursor-pointer"
          style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
        <button onClick={onConfirm} disabled={busy} className="cursor-pointer"
          style={{ display: "flex", alignItems: "center", gap: 6, background: "#EF4444", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>
          {busy && <Loader size={14} color="#fff" />}
          {busy ? "Deleting…" : "Delete"}
        </button>
      </div>
    </ModalShell>
  );
}

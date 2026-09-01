"use client";

import React from "react";
import { FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { RADIUS, SIZE, SPACE } from "@/app/(app)/memory-gap-analysis/ui";

interface DeleteTopicModalProps {
  topicLabel: string;
  questionCount: number;
  onConfirm: () => void;
  onCancel: () => void;
}

const MODAL_WIDTH = 420;

function ModalButton({ label, onClick, danger }: { label: string; onClick: () => void; danger: boolean }): React.ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        ...FONT,
        fontSize: SIZE.meta,
        fontWeight: 500,
        color: danger ? "#1e1e1c" : T.text,
        background: danger ? T.red : "transparent",
        border: `1px solid ${danger ? T.red : T.frameStrong}`,
        borderRadius: RADIUS,
        padding: "7px 14px",
        cursor: "pointer",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </button>
  );
}

/**
 * Deleting a topic is destructive to the taxonomy but never to the questions —
 * they fall back to the sink, where the next run can propose them again.
 */
export function DeleteTopicModal({ topicLabel, questionCount, onConfirm, onCancel }: DeleteTopicModalProps): React.ReactElement {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="delete-topic-title"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: SPACE.xl,
        background: "rgba(0,0,0,0.6)",
      }}
    >
      <div
        style={{
          width: MODAL_WIDTH,
          maxWidth: "100%",
          display: "flex",
          flexDirection: "column",
          gap: SPACE.lg,
          padding: SPACE.xxl,
          borderRadius: RADIUS,
          border: `1px solid ${T.frameStrong}`,
          background: T.panel,
          boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: SPACE.sm }}>
          <h2 id="delete-topic-title" style={{ ...FONT, margin: 0, fontSize: SIZE.panel, fontWeight: 600, color: T.text }}>
            Are you sure you want to delete the topic “{topicLabel}”?
          </h2>
          <p style={{ ...FONT, margin: 0, fontSize: SIZE.meta, color: T.muted, lineHeight: 1.5 }}>
            {questionCount} {questionCount === 1 ? "question moves" : "questions move"} back to Other. Nothing is lost — the next
            run can propose them as a topic again.
          </p>
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", gap: SPACE.sm }}>
          <ModalButton label="Cancel" onClick={onCancel} danger={false} />
          <ModalButton label="Delete topic" onClick={onConfirm} danger />
        </div>
      </div>
    </div>
  );
}

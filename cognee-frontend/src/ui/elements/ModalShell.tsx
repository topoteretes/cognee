"use client";

import type { ReactElement, ReactNode } from "react";
import { Modal } from "@mantine/core";

// Shared overlay + centered panel used by the dataset page modals, built on
// Mantine's Modal.Root/Overlay/Content/Body compound API so every caller gets
// real accessibility for free: focus trap, Escape-to-close, click-outside-to-
// close, role="dialog"/aria-modal, and focus restored to the trigger on
// close. Visuals are pinned to the pre-Mantine look via inline overrides;
// hex/rgba values will move to Tailwind / globals.css tokens in a later pass.
//
// The panel's background/blur/shadow live on a plain nested div rather than
// on Modal.Content itself: Mantine mirrors Modal.Content's `style` prop onto
// its full-viewport `.mantine-Modal-inner` wrapper too, which made the dark
// blurred fill cover the whole screen instead of just the panel (visible as
// a much more opaque backdrop than ShareDatasetModal's hand-rolled overlay).
// Keeping Modal.Content transparent avoids that duplication.
export default function ModalShell({
  onClose,
  width = 420,
  children,
}: {
  onClose: () => void;
  width?: number;
  children: ReactNode;
}): ReactElement {
  return (
    <Modal.Root opened onClose={onClose} size={width} centered trapFocus returnFocus closeOnEscape closeOnClickOutside>
      <Modal.Overlay style={{ background: "rgba(0,0,0,0.3)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)" }} />
      <Modal.Content style={{ background: "transparent", boxShadow: "none", padding: 0 }}>
        <div style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, boxShadow: "0 20px 60px rgba(0,0,0,0.6)" }}>
          <Modal.Body style={{ padding: 24, display: "flex", flexDirection: "column", gap: 16 }}>
            {children}
          </Modal.Body>
        </div>
      </Modal.Content>
    </Modal.Root>
  );
}

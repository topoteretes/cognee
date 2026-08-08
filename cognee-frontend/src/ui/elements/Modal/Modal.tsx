"use client";

import { useEffect } from "react";

interface ModalProps {
  isOpen: boolean;
  /** Called on Escape key press and backdrop click. Omit to disable both. */
  onClose?: () => void;
  children: React.ReactNode;
}

export default function Modal({ isOpen, onClose, children }: ModalProps) {
  // Prevent body scroll while open.
  useEffect(() => {
    if (!isOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [isOpen]);

  // Esc to close — only when a close handler is provided.
  useEffect(() => {
    if (!isOpen || !onClose) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 backdrop-blur-lg z-[10001] flex items-center justify-center"
      onClick={onClose}
    >
      {children}
    </div>
  );
}

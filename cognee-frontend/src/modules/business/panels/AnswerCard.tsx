"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import type { SessionEvent } from "../types";
import { truncate } from "../textUtils";

const QUESTION_MAX_CHARS = 90;

interface AnswerCardProps {
  event: SessionEvent | null;
  onDismiss: () => void;
}

// Ports #bv-answer (customer_tutorial.html ~7040-7060), redesigned twice on
// reviewer feedback: it now docks bottom-left, in the exact spot the
// "see what it used" notice occupied, so the eye never has to relocate
// after clicking play. Opens showing the question and the first few lines
// of the answer; clicking the card toggles the full, scrollable text.
export default function AnswerCard({ event, onDismiss }: AnswerCardProps) {
  const [expanded, setExpanded] = useState(false);

  if (!event) return null;
  const answer = typeof event.answer === "string" ? event.answer : "";
  const question = String(event.question || "");

  return (
    <div
      className="absolute bottom-12 left-3 z-10 max-w-[300px] cursor-pointer rounded-lg border px-3 py-2"
      style={{ background: "rgba(26,36,56,.9)", borderColor: "rgba(245,168,60,.4)" }}
      onClick={() => setExpanded((v) => !v)}
    >
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onDismiss(); }}
        aria-label="dismiss"
        className="absolute right-2 top-1.5 text-[#7E8CA6] hover:text-[#E9EEF6]"
      >
        ✕
      </button>
      <div className="pr-4 text-[12.5px] font-semibold text-[#F5A83C]">
        {truncate(question, QUESTION_MAX_CHARS)}
      </div>
      {answer && (
        <div
          className={`bv-answer-md mt-1 text-[12px] leading-[1.5] text-[#E9EEF6] ${
            expanded ? "max-h-[180px] overflow-y-auto" : "max-h-[56px] overflow-hidden"
          }`}
        >
          <ReactMarkdown>{answer}</ReactMarkdown>
        </div>
      )}
      {answer && !expanded && (
        <div className="mt-1 text-[11px] text-[#7E8CA6]">click to read the full answer</div>
      )}
    </div>
  );
}

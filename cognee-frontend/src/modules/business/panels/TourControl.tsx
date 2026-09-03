"use client";

import { useRef } from "react";

interface TourControlProps {
  isPlaying: boolean;
  onStart: () => void;
  onStop: () => void;
}

// A one-button flythrough trigger for useBusinessTour — new in this port,
// no source equivalent. Sits in the dock next to the LIVE indicator.
export default function TourControl({ isPlaying, onStart, onStop }: TourControlProps) {
  // BusinessView's capture-phase document pointerdown already stops the tour
  // on ANY press — including one on this button. React re-renders before the
  // click event lands on the same reused node, so a plain isPlaying-switched
  // onClick saw the post-stop state and instantly restarted the tour from
  // step 0. Remember whether the press began while playing and swallow that
  // click; keyboard activation (no pointerdown) still works via onClick.
  const pressStoppedTourRef = useRef(false);
  return (
    <button
      type="button"
      onPointerDown={() => {
        pressStoppedTourRef.current = isPlaying;
      }}
      onClick={() => {
        if (pressStoppedTourRef.current) {
          pressStoppedTourRef.current = false;
          return;
        }
        if (isPlaying) onStop();
        else onStart();
      }}
      className={`rounded-[8px] border px-2.5 py-[3px] ${
        isPlaying ? "border-[#F5A83C] text-[#F5A83C]" : "border-[#2A3652] text-[#7E8CA6] hover:text-[#E9EEF6]"
      }`}
    >
      {isPlaying ? "■ stop tour" : "▶ tour"}
    </button>
  );
}

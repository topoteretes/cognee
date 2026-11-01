"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import FeedbackModal from "@/ui/layout/FeedbackModal";
import isCloudEnvironment from "@/utils/isCloudEnvironment";

// `?replay=1` tells the route guard this is a deliberate re-entry — without it
// an already-onboarded user is bounced straight back to /dashboard.
const ONBOARDING_LINK = "/onboarding?replay=1";
// Same host that gates Segment tracking (modules/analytics/isProductionHost) —
// that helper reads next/headers, so this client component checks it itself.
const PRODUCTION_HOST = "platform.cognee.ai";
// HubSpot Meetings, not Calendly (compliance item A3.4): the privacy notice
// lists HubSpot as the only booking processor, so a Calendly link here would
// send the user's IP to a US processor the notice never mentions. Same
// calendar the in-app Book demo modal embeds (BookingForm.tsx), which is why
// no new vendor or CSP entry comes with it.
const BOOK_A_CALL_LINK = "https://meetings-eu1.hubspot.com/luca-stromann";

// Same resting colour as the main nav items above: at 35% these read as
// disabled, and people did not see them at all.
const ROW_COLOR = "rgba(237,236,234,0.7)";

const ROW_STYLE: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 7,
  width: "100%",
  padding: "7px 12px",
  background: "transparent",
  border: "none",
  borderRadius: 8,
  fontFamily: "inherit",
  fontSize: 12.5,
  fontWeight: 500,
  color: ROW_COLOR,
  textDecoration: "none",
  transition: "color 120ms ease, background 120ms ease",
};

function hoverIn(e: React.MouseEvent<HTMLElement>): void {
  e.currentTarget.style.color = "#EDECEA";
  e.currentTarget.style.background = "rgba(255,255,255,0.06)";
}

function hoverOut(e: React.MouseEvent<HTMLElement>): void {
  e.currentTarget.style.color = ROW_COLOR;
  e.currentTarget.style.background = "transparent";
}

function CalendarIcon(): React.JSX.Element {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.7 }}>
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
      <path d="M10 14l2 2 4-4" />
    </svg>
  );
}

function FeedbackIcon(): React.JSX.Element {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.7 }}>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function OnboardingIcon(): React.JSX.Element {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.7 }}>
      <path d="M9 11l3 3L22 4" />
      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
    </svg>
  );
}

interface NavbarFooterProps {
  railed: boolean;
}

// The local app has no onboarding or Cloud support actions.
export default function NavbarFooter({ railed }: NavbarFooterProps): React.JSX.Element | null {
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  // Onboarding replay is hidden on production for now. Assume production until
  // the host is known: SSR and first client paint agree (no hydration
  // mismatch), and the button never flashes at platform.cognee.ai — it only
  // ever appears, after mount, everywhere else.
  const [isProductionHost, setIsProductionHost] = useState(true);

  useEffect(() => {
    setIsProductionHost(window.location.hostname === PRODUCTION_HOST);
  }, []);

  if (!isCloudEnvironment()) return null;

  return (
    <div style={{ padding: 12, borderTop: "1px solid rgba(255,255,255,0.08)", display: "flex", flexDirection: "column", gap: 4 }}>
      {isCloudEnvironment() && (
        <>
          <Link
            href={BOOK_A_CALL_LINK}
            target="_blank"
            rel="noopener noreferrer"
            title={railed ? "Book a call" : undefined}
            className="cursor-pointer"
            style={ROW_STYLE}
            onMouseEnter={hoverIn}
            onMouseLeave={hoverOut}
          >
            <CalendarIcon />
            {!railed && "Book a call"}
          </Link>
          <button
            onClick={() => setFeedbackOpen(true)}
            title={railed ? "Feedback" : undefined}
            className="cursor-pointer"
            style={ROW_STYLE}
            onMouseEnter={hoverIn}
            onMouseLeave={hoverOut}
          >
            <FeedbackIcon />
            {!railed && "Feedback"}
          </button>
        </>
      )}
      {!isProductionHost && (
        <Link
          href={ONBOARDING_LINK}
          title={railed ? "Onboarding" : undefined}
          className="cursor-pointer"
          style={ROW_STYLE}
          onMouseEnter={hoverIn}
          onMouseLeave={hoverOut}
        >
          <OnboardingIcon />
          {!railed && "Onboarding"}
        </Link>
      )}
      {feedbackOpen && <FeedbackModal onClose={() => setFeedbackOpen(false)} />}
    </div>
  );
}

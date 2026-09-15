// Hand-drawn illustration for the auto-recharge announcement — a credit card
// with a refresh loop and rising sparks, in the same lavender/violet palette
// as AutoRechargePanel (ACCENT #BC9BFF / PRIMARY #6510F4). Not a product
// screenshot: this modal introduces a settings toggle that has no visual
// identity of its own worth photographing, so a themed icon-scene reads
// better than a boring form screenshot.
export default function AutoRechargeIllustration(): React.ReactElement {
  return (
    <svg
      viewBox="0 0 320 320"
      width="100%"
      height="100%"
      style={{ display: "block" }}
      role="img"
      aria-label="Illustration of a credit card automatically recharging"
    >
      <defs>
        <radialGradient id="arGlow" cx="50%" cy="42%" r="60%">
          <stop offset="0%" stopColor="#BC9BFF" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#BC9BFF" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="arCard" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#8B5CF6" />
          <stop offset="100%" stopColor="#6510F4" />
        </linearGradient>
      </defs>

      <circle cx="160" cy="150" r="140" fill="url(#arGlow)" />

      {/* card */}
      <g transform="translate(70,110) rotate(-8)">
        <rect x="0" y="0" width="180" height="112" rx="16" fill="url(#arCard)" />
        <rect x="0" y="30" width="180" height="18" fill="rgba(0,0,0,0.25)" />
        <rect x="16" y="66" width="46" height="10" rx="5" fill="rgba(255,255,255,0.55)" />
        <rect x="16" y="84" width="80" height="8" rx="4" fill="rgba(255,255,255,0.3)" />
      </g>

      {/* refresh loop badge */}
      <g transform="translate(196,86)">
        <circle cx="0" cy="0" r="34" fill="#1e1e1c" stroke="#BC9BFF" strokeWidth="2" />
        <path
          d="M -14 -2 A 14 14 0 1 1 -12 10"
          fill="none"
          stroke="#BC9BFF"
          strokeWidth="4"
          strokeLinecap="round"
        />
        <path d="M -12 10 L -18 6 L -6 2 Z" fill="#BC9BFF" />
      </g>

      {/* sparks rising to signal a top-up just landed */}
      <g stroke="#D9C7FF" strokeWidth="3" strokeLinecap="round">
        <line x1="120" y1="60" x2="120" y2="44" />
        <line x1="140" y1="70" x2="150" y2="52" />
        <line x1="100" y1="72" x2="90" y2="56" />
      </g>
    </svg>
  );
}

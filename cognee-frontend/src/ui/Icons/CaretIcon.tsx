export default function CaretIcon({ width = 50, height = 36, color = "currentColor", className = "" }) {
  return (
    <svg width={width} height={height} viewBox="0 0 50 36" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
      <path d="M4 32L25 5" stroke={color} strokeWidth="8" strokeLinecap="round"/>
      <path d="M46 32L25 5" stroke={color} strokeWidth="8" strokeLinecap="round"/>
    </svg>
  );
}

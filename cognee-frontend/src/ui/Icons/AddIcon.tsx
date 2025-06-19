export default function SearchIcon({ width = 24, height = 24, color = 'currentColor', className = '' }) {
  return (
    <svg width={width} height={height} viewBox="0 0 50 50" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
      <path d="M24.9999 46L24.9999 4M46.0049 25.005L4.00488 25.005" stroke={color} strokeWidth="8" strokeLinecap="round"/>
    </svg>
  );
}

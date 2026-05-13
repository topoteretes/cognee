export default function PlayIcon({ width = 11, height = 14, color = "#000000", className = "" }) {
  return (
    <svg className={className} width={width} height={height} viewBox="0 0 11 14" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M1 1L10.3333 7L1 13V1Z" stroke={color} strokeWidth="1.33" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

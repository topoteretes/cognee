export default function PlusIcon({ width = 16, height = 16, color = "#000000", className = "" }) {
  return (
    <svg className={className} width={width} height={height} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M4.09637 8H12.8675" stroke={color} strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M8.48193 3.33331V12.6666" stroke={color} strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

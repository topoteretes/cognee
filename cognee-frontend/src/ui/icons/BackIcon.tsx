export default function BackIcon({ width = 16, height = 16, color = "#17191C", className = "" }) {
  return (
    <svg className={className} width={width} height={height} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M7.99992 12.6666L3.33325 7.99998L7.99992 3.33331" stroke={color} strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12.6666 8H3.33325" stroke={color} strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

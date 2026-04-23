export default function CopyIcon({ width = "28", height = "35", className = "", color = "currentColor" }) {
  return (
    <svg width={width} height={height} viewBox="0 0 28 35" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
      <path d="M0 8.5C0 7.94771 0.447715 7.5 1 7.5H17C19.2091 7.5 21 9.29086 21 11.5V33.5C21 34.0523 20.5523 34.5 20 34.5H1C0.447715 34.5 0 34.0523 0 33.5V8.5Z" fill={color}/>
        <line x1="5" y1="13" x2="15" y2="13" stroke="#1E1E1E"/>
        <line x1="5" y1="16" x2="15" y2="16" stroke="#1E1E1E"/>
        <line x1="5" y1="19" x2="15" y2="19" stroke="#1E1E1E"/>
        <line x1="5" y1="22" x2="15" y2="22" stroke="#1E1E1E"/>
        <line x1="5" y1="25" x2="12" y2="25" stroke="#1E1E1E"/>
      <path d="M23 1C25.4853 1 27.5 3.01472 27.5 5.5V27.5C27.5 28.3284 26.8284 29 26 29H7C6.17157 29 5.5 28.3284 5.5 27.5V2.5C5.5 1.67157 6.17157 1 7 1H23Z" fill={color} stroke="#1E1E1E"/>
      <path d="M11 7H22" stroke="#1E1E1E" strokeLinecap="round"/>
      <path d="M11 10H22" stroke="#1E1E1E" strokeLinecap="round"/>
      <path d="M11 13H22" stroke="#1E1E1E" strokeLinecap="round"/>
      <path d="M11 16H22" stroke="#1E1E1E" strokeLinecap="round"/>
      <path d="M11 19H18" stroke="#1E1E1E" strokeLinecap="round"/>
    </svg>

  );
}

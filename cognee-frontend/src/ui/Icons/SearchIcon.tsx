export default function SearchIcon({ width = 12, height = 12, color = "#D8D8D8", className = "" }) {
  return (
    <svg className={className} width={width} height={height} viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M5.5 9.5C7.70914 9.5 9.5 7.70914 9.5 5.5C9.5 3.29086 7.70914 1.5 5.5 1.5C3.29086 1.5 1.5 3.29086 1.5 5.5C1.5 7.70914 3.29086 9.5 5.5 9.5Z" stroke={color} strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M10.5 10.5L8.35001 8.34998" stroke={color} strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

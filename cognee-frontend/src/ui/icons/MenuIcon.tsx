export default function AddIcon({ width = 16, height = 16, color = "#000000", className = "" }) {
  return (
    <svg className={className} width={width} height={height} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="8" cy="4" r="1" fill={color} />
      <circle cx="8" cy="8" r="1" fill={color} />
      <circle cx="8" cy="12" r="1" fill={color} />
    </svg>
  );
}

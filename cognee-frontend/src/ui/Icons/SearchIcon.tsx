export default function SearchIcon({ width = 24, height = 24, color = 'currentColor', className = '' }) {
  return (
    <svg width={width} height={height} viewBox="0 0 50 50" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
      <circle cx="19.5" cy="19.5" r="17" stroke={color} strokeWidth="5"/>
      <path d="M8 19.5C8 13.1487 13.1487 8 19.5 8" stroke={color}/>
      <path d="M43.2782 48.9312C44.897 50.4344 47.428 50.3406 48.9312 48.7218C50.4344 47.103 50.3406 44.572 48.7218 43.0688L43.2782 48.9312ZM46 46L48.7218 43.0688L34.7218 30.0688L32 33L29.2782 35.9312L43.2782 48.9312L46 46Z" fill={color}/>
    </svg>
  );
}

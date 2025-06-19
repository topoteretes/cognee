import { NodeObject } from "react-force-graph-2d";
import getColorForNodeType from './getColorForNodeType';

interface GraphLegendProps {
  data?: NodeObject[];
}

export default function GraphLegend({ data }: GraphLegendProps) {
  const legend: Set<string> = new Set();

  for (let i = 0; i < Math.min(data?.length || 0, 100); i++) {
    legend.add(data![i].type);
  }

  return (
    <div className="flex flex-col gap-1">
      {Array.from(legend).map((nodeType) => (
        <div key={nodeType} className="flex flex-row items-center gap-2">
          <span className="w-2 h-2 rounded-2xl" style={{ backgroundColor: getColorForNodeType(nodeType) }} />
          <span className="text-white">{nodeType}</span>
        </div>
      ))}
    </div>
  );
}

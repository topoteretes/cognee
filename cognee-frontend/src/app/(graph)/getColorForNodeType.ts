import colors from "tailwindcss/colors";
import { formatHex } from "culori";

const NODE_COLORS = {
  TextDocument: formatHex(colors.stone[200]),
  DocumentChunk: formatHex(colors.stone[300]),
  TextSummary: formatHex(colors.blue[300]),
  Entity: formatHex(colors.indigo[300]),
  EntityType: formatHex(colors.indigo[400]),
  NodeSet: formatHex(colors.indigo[400]),
  GitHubUser: formatHex(colors.gray[200]),
  Comment: formatHex(colors.blue[300]),
  Issue: formatHex(colors.red[200]),
  Repository: formatHex(colors.stone[200]),
  Commit: formatHex(colors.teal[300]),
  File: formatHex(colors.emerald[300]),
  FileChange: formatHex(colors.sky[300]),
};

export default function getColorForNodeType(type: string) {
  return NODE_COLORS[type as keyof typeof NODE_COLORS] || colors.gray[500];
}

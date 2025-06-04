import colors from "tailwindcss/colors";
import { formatHex } from "culori";

const NODE_COLORS = {
  Document: formatHex(colors.blue[500]),
  Chunk: formatHex(colors.green[500]),
  Entity: formatHex(colors.yellow[500]),
  EntityType: formatHex(colors.purple[500]),
  NodeSet: formatHex(colors.indigo[800]),
  GitHubUser: formatHex(colors.gray[300]),
  Comment: formatHex(colors.amber[500]),
  Issue: formatHex(colors.red[500]),
  Repository: formatHex(colors.stone[400]),
  Commit: formatHex(colors.teal[500]),
  File: formatHex(colors.emerald[500]),
  FileChange: formatHex(colors.sky[500]),
};

export default function getColorForNodeType(type: string) {
  return NODE_COLORS[type as keyof typeof NODE_COLORS] || colors.gray[500];
}

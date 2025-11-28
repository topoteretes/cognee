export interface Node {
  id: string;
  label: string;
  type: string;
}

export interface Edge {
  id: string;
  label: string;
  source: string;
  target: string;
}

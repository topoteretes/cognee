export interface Cell {
  id: string;
  name: string;
  type: "markdown" | "code";
  content: string;
  result?: [];
  error?: string;
}

export interface Notebook {
  id: string;
  name: string;
  cells: Cell[];
}

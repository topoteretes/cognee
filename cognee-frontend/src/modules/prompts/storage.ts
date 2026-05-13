import { v4 as uuid } from "uuid";

export interface Prompt {
  id: string;
  name: string;
  content: string;
  createdAt: string;
  updatedAt: string;
}

const STORAGE_KEY = "cognee-prompts";

export const DEFAULT_PROMPT_TEMPLATE = `You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph from technical and scientific content.
**Nodes** represent technologies, algorithms, protocols, data structures, scientific concepts, and specifications.
**Edges** represent relationships between technical entities such as dependencies, implementations, and comparisons.
The aim is to achieve precision and traceability in the knowledge graph.
# 1. Labeling Nodes
**Consistency**: Use fundamental technical categories for node labels.
  - For example, when you identify a programming language, always label it as **"Technology"**.
  - When you identify a method or procedure, label it as **"Algorithm"**.
  - Avoid overly specific labels like "SortingAlgorithm" or "NoSQLDatabase" — keep those as properties.
  - Don't use too generic terms like "Entity" or "Thing".
**Node IDs**: Never utilize integers as node IDs.
  - Node IDs should be the canonical name of the technology, concept, or specification (e.g., "TCP/IP", "QuickSort", "PostgreSQL").
# 2. Handling Versions, Metrics, and Dates
  - When you identify a version number, attach it as a **"version"** property on the relevant node.
  - Extract performance metrics, benchmarks, or measurements as key-value properties.
  - For dates (release dates, publication dates), use the format "YYYY-MM-DD" where possible.
  - **Property Format**: Properties must be in a key-value format.
  - **Quotation Marks**: Never use escaped single or double quotes within property values.
  - **Naming Convention**: Use snake_case for relationship names, e.g., \`depends_on\`, \`implements\`, \`extends\`, \`compares_to\`, \`supersedes\`.
# 3. Coreference Resolution
  - **Maintain Entity Consistency**: Technologies and concepts are often referred to by abbreviations, acronyms, or informal names.
  Always resolve these to the most complete and canonical identifier (e.g., "JS" → "JavaScript", "k8s" → "Kubernetes").
  - If a concept appears under multiple names, unify them under a single node.
# 4. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.`;

export function loadPrompts(): Prompt[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Prompt[]) : [];
  } catch {
    return [];
  }
}

export function savePrompts(prompts: Prompt[]): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(prompts));
}

export function createPrompt(name: string): Prompt {
  const now = new Date().toISOString();
  return {
    id: uuid(),
    name,
    content: DEFAULT_PROMPT_TEMPLATE,
    createdAt: now,
    updatedAt: now,
  };
}

export function getPrompt(id: string): Prompt | undefined {
  return loadPrompts().find((p) => p.id === id);
}

export function upsertPrompt(prompt: Prompt): void {
  const prompts = loadPrompts();
  const idx = prompts.findIndex((p) => p.id === prompt.id);
  const updated = { ...prompt, updatedAt: new Date().toISOString() };
  if (idx >= 0) {
    prompts[idx] = updated;
  } else {
    prompts.push(updated);
  }
  savePrompts(prompts);
}

export function deletePrompt(id: string): void {
  savePrompts(loadPrompts().filter((p) => p.id !== id));
}

import { CogneeInstance } from "@/modules/instances/types";

export interface OntologyMeta {
  filename: string;
  size_bytes: number;
  uploaded_at: string;
  description?: string;
}

/** Returns a dict mapping ontology_key → metadata */
export async function listOntologies(
  instance: CogneeInstance,
): Promise<Record<string, OntologyMeta>> {
  const resp = await instance.fetch("/v1/ontologies");
  if (!resp.ok) return {};
  return resp.json();
}

export async function deleteOntology(
  instance: CogneeInstance,
  ontologyKey: string,
): Promise<void> {
  const resp = await instance.fetch(`/v1/ontologies/${encodeURIComponent(ontologyKey)}`, {
    method: "DELETE",
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || `Delete failed: ${resp.status}`);
  }
}

export async function uploadOntology(
  instance: CogneeInstance,
  ontologyKey: string,
  file: File,
  description?: string,
): Promise<void> {
  const formData = new FormData();
  formData.append("ontology_key", ontologyKey);
  formData.append("ontology_file", file);
  if (description) {
    formData.append("description", description);
  }
  const resp = await instance.fetch("/v1/ontologies", {
    method: "POST",
    body: formData,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || `Upload failed: ${resp.status}`);
  }
}

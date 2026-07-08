import { CogneeInstance } from "../instances/types";

export interface LLMSettings {
  apiKey: string | null;
  provider: string;
  model: string;
}

export function getLLMSettings(instance: CogneeInstance): Promise<LLMSettings> {
  return instance.fetch("/v1/settings", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((settings) => settings.llm);
}

export function saveLLMApiKey(
  instance: CogneeInstance,
  { provider, model, apiKey }: { provider: string; model: string; apiKey: string },
) {
  return instance.fetch("/v1/settings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ llm: { provider, model, apiKey } }),
  });
}

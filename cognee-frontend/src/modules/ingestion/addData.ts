import { CogneeInstance } from "../instances/types";

export default async function addData(dataset: { id?: string, name?: string }, files: File[], instance: CogneeInstance) {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("data", file, file.name);
  });
  if (dataset.id) {
    formData.append("datasetId", dataset.id);
  }
  if (dataset.name) {
    formData.append("datasetName", dataset.name);
  }

  return instance.fetch("/v1/add", {
    method: "POST",
    body: formData,
  }).then((response) => response.json());
}

export async function addUrlData(dataset: { id?: string, name?: string }, url: string, instance: CogneeInstance) {
  const data = {
    textData: [url],
    datasetId: dataset.id,
    datasetName: dataset.name,
  };

  return instance.fetch("/v1/add_text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  }).then((response) => response.json());
}

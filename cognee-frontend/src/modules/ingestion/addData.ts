import { fetch } from "@/utils";

export default function addData(dataset: { id?: string, name?: string }, files: File[]) {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("data", file, file.name);
  })
  if (dataset.id) {
    formData.append("datasetId", dataset.id);
  }
  if (dataset.name) {
    formData.append("datasetName", dataset.name);
  }

  return fetch("/v1/add", {
    method: "POST",
    body: formData,
  }).then((response) => response.json());
}

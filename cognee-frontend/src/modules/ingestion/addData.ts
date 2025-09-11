import { fetch } from "@/utils";

export default async function addData(dataset: { id?: string, name?: string }, files: File[], useCloud = false) {
  if (useCloud) {
    const data = {
      text_data: await Promise.all(files.map(async (file) => file.text())),
      datasetId: dataset.id,
      datasetName: dataset.name,
    };

    return fetch("/v1/add", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    }, true).then((response) => response.json());
  } else {
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
}

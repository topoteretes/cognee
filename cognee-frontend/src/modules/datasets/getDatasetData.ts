export default function getDatasetData(dataset: { id: string }) {
  return fetch(`http://127.0.0.1:8000/datasets/${dataset.id}/data`)
      .then((response) => response.json());
}

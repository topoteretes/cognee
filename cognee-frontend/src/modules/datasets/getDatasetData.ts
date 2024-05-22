export default function getDatasetData(dataset: { id: string }) {
  return fetch(`http://0.0.0.0:8000/datasets/${dataset.id}/data`)
      .then((response) => response.json());
}

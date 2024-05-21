export default function deleteDataset(dataset: { id: string }) {
  return fetch(`http://0.0.0.0:8000/datasets/${dataset.id}`, {
    method: 'DELETE',
  })
}

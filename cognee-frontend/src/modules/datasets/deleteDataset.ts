export default function deleteDataset(dataset: { id: string }) {
  return fetch(`http://127.0.0.1:8000/datasets/${dataset.id}`, {
    method: 'DELETE',
  })
}

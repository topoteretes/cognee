export default function addData(dataset: { id: string }, files: File[]) {
  const formData = new FormData();
  formData.append('datasetId', dataset.id);
  const file = files[0];
  formData.append('data', file, file.name);

  return fetch('http://0.0.0.0:8000/add', {
    method: 'POST',
    body: formData,
  }).then((response) => response.json());
}

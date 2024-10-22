export default function handleServerErrors(response) {
  return new Promise((resolve, reject) => {
    if (response.status === 401) {
      return reject(new Error('Unauthorized'));
    }
    if (!response.ok) {
      if (response.json) {
        return response.json().then(error => reject(error));
      } else {
        return reject(response.detail || response.body || response);
      }
    }

    return resolve(response);
  });
}

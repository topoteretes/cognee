export default function handleServerErrors(response: Response): Promise<Response> {
  return new Promise((resolve, reject) => {
    if (response.status === 401) {
      window.location.href = '/auth';
      return;
    }
    if (!response.ok) {
      return response.json().then(error => reject(error));
    }

    return resolve(response);
  });
}

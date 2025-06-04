import handleServerErrors from "./handleServerErrors";

let numberOfRetries = 0;

export default async function fetch(url: string, options: RequestInit = {}): Promise<Response> {
  function retry(lastError: Response) {
    if (numberOfRetries > 1) {
      return Promise.reject(lastError);
    }

    numberOfRetries += 1;

    return window.fetch("/auth/token")
      .then(() => {
        return fetch(url, options);
      });
  }

  return global.fetch("http://localhost:8000/api" + url, {
    ...options,
    credentials: "include",
  })
    .then((response) => handleServerErrors(response, retry))
    .then((response) => {
      numberOfRetries = 0;

      return response;
    })
    .catch((error) => {
      if (error.status === 401) {
        return retry(error);
      }
      return Promise.reject(error);
    });
}

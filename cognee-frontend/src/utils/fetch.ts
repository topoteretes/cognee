import handleServerErrors from "./handleServerErrors";

let numberOfRetries = 0;

const isAuth0Enabled = process.env.USE_AUTH0_AUTHORIZATION?.toLowerCase() === "true"

export default async function fetch(url: string, options: RequestInit = {}): Promise<Response> {
  function retry(lastError: Response) {
    if (!isAuth0Enabled) {
      return Promise.reject(lastError);
    }

    if (numberOfRetries >= 1) {
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
      if (error.detail === undefined) {
        return Promise.reject(
          new Error("No connection to the server.")
        );
      }

      if (error.status === 401) {
        return retry(error);
      }
      return Promise.reject(error);
    });
}

import handleServerErrors from "./handleServerErrors";
import isCloudEnvironment from "./isCloudEnvironment";

let numberOfRetries = 0;

const isAuth0Enabled = process.env.USE_AUTH0_AUTHORIZATION?.toLowerCase() === "true";

const backendApiUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8000/api";

const cloudApiUrl = process.env.NEXT_PUBLIC_CLOUD_API_URL || "http://localhost:8001/api";

let apiKey: string | null = null;
let accessToken: string | null = null;

export default async function fetch(url: string, options: RequestInit = {}, useCloud = false): Promise<Response> {
  function retry(lastError: Response) {
    if (!isAuth0Enabled) {
      return Promise.reject(lastError);
    }

    if (numberOfRetries >= 1) {
      return Promise.reject(lastError);
    }

    numberOfRetries += 1;

    return global.fetch("/auth/token")
      .then(() => {
        return fetch(url, options);
      });
  }

  return global.fetch(
    (useCloud ? cloudApiUrl : backendApiUrl) + (useCloud ? url.replace("/v1", "") : url),
    {
      ...options,
      headers: {
        ...options.headers,
        ...(useCloud && !isCloudEnvironment()
          ? {"X-Api-Key": apiKey!}
          : {"Authorization": `Bearer ${accessToken}`}
        ),
      },
      credentials: "include",
    },
  )
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

fetch.checkHealth = () => {
  return global.fetch(`${backendApiUrl.replace("/api", "")}/health`);
};

fetch.setApiKey = (newApiKey: string) => {
  apiKey = newApiKey;
};

fetch.setAccessToken = (newAccessToken: string) => {
  accessToken = newAccessToken;
};

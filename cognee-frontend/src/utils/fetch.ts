import handleServerErrors from "./handleServerErrors";
import isCloudEnvironment from "./isCloudEnvironment";

let numberOfRetries = 0;

const isAuth0Enabled = process.env.USE_AUTH0_AUTHORIZATION?.toLowerCase() === "true";

const backendApiUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8000";

const cloudApiUrl = process.env.NEXT_PUBLIC_CLOUD_API_URL || "http://localhost:8001";

let apiKey: string | null = process.env.NEXT_PUBLIC_COGWIT_API_KEY || null;
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

  const authHeaders = useCloud && (!isCloudEnvironment() || !accessToken) ? {
    "X-Api-Key": apiKey,
  } : {
    "Authorization": `Bearer ${accessToken}`,
  }

  return global.fetch(
    (useCloud ? cloudApiUrl : backendApiUrl) + "/api" + (useCloud ? url.replace("/v1", "") : url),
    {
      ...options,
      headers: {
        ...options.headers,
        ...authHeaders,
      } as HeadersInit,
      credentials: "include",
    },
  )
    .then((response) => handleServerErrors(response, retry, useCloud))
    .catch((error) => {
      if (error.detail === undefined) {
        return Promise.reject(
          new Error("No connection to the server.")
        );
      }

      return Promise.reject(error);
    })
    .finally(() => {
      numberOfRetries = 0;
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

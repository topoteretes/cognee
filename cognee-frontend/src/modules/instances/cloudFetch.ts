import handleServerErrors from "@/utils/handleServerErrors";

// let numberOfRetries = 0;

const cloudApiUrl = process.env.NEXT_PUBLIC_CLOUD_API_URL || "http://localhost:8001";

let apiKey: string | null = process.env.NEXT_PUBLIC_COGWIT_API_KEY || null;

export function setApiKey(newApiKey: string) {
  apiKey = newApiKey;
};

export default async function cloudFetch(url: URL | RequestInfo, options: RequestInit = {}): Promise<Response> {
  // function retry(lastError: Response) {
  //   if (numberOfRetries >= 1) {
  //     return Promise.reject(lastError);
  //   }

  //   numberOfRetries += 1;

  //   return global.fetch("/auth/token")
  //     .then(() => {
  //       return fetch(url, options);
  //     });
  // }

  const authHeaders = {
    "Authorization": `X-Api-Key ${apiKey}`,
  };

  return global.fetch(
    cloudApiUrl + "/api" + (typeof url === "string" ? url : url.toString()).replace("/v1", ""),
    {
      ...options,
      headers: {
        ...options.headers,
        ...authHeaders,
      } as HeadersInit,
      credentials: "include",
    },
  )
    .then((response) => handleServerErrors(response, null, true))
    .catch((error) => {
      if (error.message === "NEXT_REDIRECT") {
        throw error;
      }

      if (error.detail === undefined) {
        return Promise.reject(
          new Error("No connection to the server.")
        );
      }

      return Promise.reject(error);
    });
    // .finally(() => {
    //   numberOfRetries = 0;
    // });
}

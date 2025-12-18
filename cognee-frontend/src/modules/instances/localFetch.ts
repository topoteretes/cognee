import handleServerErrors from "@/utils/handleServerErrors";

const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || "http://localhost:8000";

export default async function localFetch(url: URL | RequestInfo, options: RequestInit = {}): Promise<Response> {
  return global.fetch(
    localApiUrl + "/api" + url,
    {
      ...options,
      credentials: "include",
    },
  )
    .then((response) => handleServerErrors(response, null, false))
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
}

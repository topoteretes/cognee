import handleServerErrors from "./handleServerErrors";

export default async function fetch(url: string, options: RequestInit = {}): Promise<Response> {
  return global.fetch("http://localhost:8000/api" + url, {
    ...options,
    credentials: "include",
  })
    .then(handleServerErrors);
}

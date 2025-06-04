import { redirect } from "next/navigation";

export default function handleServerErrors(response: Response, retry?: (response: Response) => Promise<Response>): Promise<Response> {
  return new Promise((resolve, reject) => {
    if (response.status === 401) {
      if (retry) {
        return retry(response);
      } else {
        return redirect("/auth");
      }
    }
    if (!response.ok) {
      return response.json().then(error => reject(error));
    }

    return resolve(response);
  });
}

import { redirect } from "next/navigation";

export default function handleServerErrors(response: Response, retry?: (response: Response) => Promise<Response>): Promise<Response> {
  return new Promise((resolve, reject) => {
    if (response.status === 401) {
      if (retry) {
        return retry(response)
          .catch(() => {
            return redirect("/auth/login");
          });
      } else {
        return redirect("/auth/login");
      }
    }
    if (!response.ok) {
      return response.json().then(error => reject(error));
    }

    if (response.status >= 200 && response.status < 300) {
      return resolve(response);
    }

    return reject(response);
  });
}

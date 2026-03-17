import { redirect } from "next/navigation";

export default function handleServerErrors(
  response: Response,
  retry: ((response: Response) => Promise<Response>) | null = null,
  useCloud: boolean = false,
): Promise<Response> {
  return new Promise((resolve, reject) => {
    if ((response.status === 401 || response.status === 403) && !useCloud) {
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
      return response.json().then(error => {
        error.status = response.status;
        reject(error);
      });
    }

    if (response.status >= 200 && response.status < 300) {
      return resolve(response);
    }

    return reject(response);
  });
}

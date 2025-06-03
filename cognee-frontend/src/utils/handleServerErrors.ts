import { redirect } from "next/navigation";

export default function handleServerErrors(response: Response): Promise<Response> {
  return new Promise((resolve, reject) => {
    if (response.status === 401) {
      return redirect("/auth");
    }
    if (!response.ok) {
      return response.json().then(error => reject(error));
    }

    return resolve(response);
  });
}

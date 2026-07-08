import { redirect } from "next/navigation";

export default function handleServerErrors(
  response: Response,
  retry: ((response: Response) => Promise<Response>) | null = null,
  useCloud: boolean = true,
): Promise<Response> {
  return new Promise((resolve, reject) => {
    // Local mode: redirect to local login on auth failures
    if ((response.status === 401 || response.status === 403) && !useCloud) {
      if (typeof window !== "undefined") {
        window.location.href = "/local-login";
      }
      return reject(new Error("Session expired"));
    }
    if ((response.status === 401 || response.status === 403) && useCloud) {
      // 403 = authenticated but not authorized for this resource.
      // Only redirect for email-verification; otherwise reject so callers
      // can handle the error gracefully (avoids redirect loops).
      if (response.status === 403) {
        return response.clone().text().then((text) => {
          if (text.toLowerCase().includes("verify your email")) {
            return redirect("/verify-email");
          }
          const error: Record<string, unknown> = { message: text || "Forbidden", status: 403, statusText: "Forbidden" };
          reject(error);
        });
      }
      // 401 = not authenticated — redirect to sign-in
      if (retry) {
        return retry(response)
          .catch(() => {
            return redirect("/sign-in");
          });
      } else {
        return redirect("/sign-in");
      }
    }
    if (!response.ok) {
      return response.text().then(text => {
        let error: Record<string, unknown> = {};
        try {
          error = JSON.parse(text);
        } catch {
          error = { message: text || response.statusText };
        }
        error.status = response.status;
        error.statusText = response.statusText;
        reject(error);
      });
    }

    if (response.status >= 200 && response.status < 300) {
      return resolve(response);
    }

    return reject(response);
  });
}

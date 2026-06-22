import { redirect } from "next/navigation";

// Build a real Error (not a bare object) from a server error response.
//
// Rejecting with a plain object is why callers logged "ERROR: {}" — a plain
// object has no useful default serialization in console.error. An Error
// serializes with its message (and stack). The server's parsed JSON fields
// (detail/error/message/...) are copied onto the Error so existing callers that
// read `error.detail`/`error.error`/`error.message` keep working.
function buildServerError(body: string, status: number, statusText: string): Error {
  let parsed: Record<string, unknown> = {};
  if (body) {
    try {
      parsed = JSON.parse(body);
    } catch {
      parsed = { message: body };
    }
  }
  const detail = parsed.detail ?? parsed.error ?? parsed.message;
  const message =
    typeof detail === "string" && detail.trim()
      ? detail
      : (body && body.trim()) || statusText || `Request failed with status ${status}`;
  const error = Object.assign(new Error(message), parsed, { status, statusText });
  // Ensure the resolved message wins even if `parsed` carried its own `message`.
  error.message = message;
  return error;
}

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
          reject(buildServerError(text || "Forbidden", 403, "Forbidden"));
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
        reject(buildServerError(text, response.status, response.statusText));
      });
    }

    if (response.status >= 200 && response.status < 300) {
      return resolve(response);
    }

    return reject(buildServerError("", response.status, response.statusText));
  });
}

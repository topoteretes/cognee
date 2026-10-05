import { HttpError, isApiErrorBody } from "@/services/http/errors";

// HTTP 402 is raised only by the pod's pre-flight credit guard on
// cognify/remember/search/recall/improve, so status alone reliably identifies it.
export function isInsufficientCreditsError(error: unknown): error is HttpError {
  return error instanceof HttpError && error.status === 402;
}

// CLO-305 added a structured field to the 402 body: {detail, reason,
// operation, remaining_usd}. Prefer reading it directly; fall back to
// regex-parsing the free-text detail string (the only thing available
// before that pod change ships) so this keeps working against an
// un-upgraded pod without throwing.
export function parseInsufficientCreditsOperation(error: HttpError): string | null {
  if (isApiErrorBody(error.body) && typeof error.body.operation === "string") {
    return error.body.operation;
  }
  const match = /run (\w+)/.exec(error.message);
  return match ? match[1] : null;
}

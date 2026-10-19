"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { request, describeError, type WorkspaceContext } from "@/modules/workspace/api";
import Sources from "../../workspace/partials/Sources";

export default function LocalSources() {
  const [context, setContext] = useState<WorkspaceContext | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let cancelled = false;
    request<WorkspaceContext>("/v1/workspace/context")
      .then((value) => { if (!cancelled) setContext(value); })
      .catch((error) => { if (!cancelled) setError(describeError(error)); });
    return () => { cancelled = true; };
  }, []);
  return <section><h2>Data sources on this server</h2>
    <p>Connections belong to your account. Share their datasets through <Link href="/workspace">Manage memory</Link>.</p>
    {error && <p role="alert">{error}</p>}{context && <Sources context={context} />}
  </section>;
}

"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { getLocalApiUrl } from "@/modules/users/getLocalApiUrl";
import { request, describeError, type WorkspaceContext } from "@/modules/workspace/api";
import Sources from "./partials/Sources";
import Access from "./partials/Access";
import Promote from "./partials/Promote";
import Team from "./partials/Team";
import Activity from "./partials/Activity";
import styles from "./workspace.module.css";

const TABS = ["Sources", "Agent access", "Promote memory", "Team", "Activity"] as const;

export default function WorkspacePage() {
  const [context, setContext] = useState<WorkspaceContext | null>(null);
  const [tab, setTab] = useState<typeof TABS[number]>("Sources");
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    try { setContext(await request<WorkspaceContext>("/v1/workspace/context")); setError(""); }
    catch (error) { setError(describeError(error)); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  return <div className={styles.page}>
    <div className={styles.header}>
      <div><h1>Manage memory</h1><p>Connect sources. Set agent access. Review what becomes shared memory.</p>
        <div className={styles.muted}>{getLocalApiUrl()} · {context?.user.email ?? "Checking your account…"}</div></div>
      <div className={styles.row}><Link className={styles.button} href="/sessions">Inspect sessions</Link>
        <button className={styles.button} onClick={() => setTab("Activity")}>Live agent activity</button></div>
    </div>
    {error && <div role="alert" className={styles.error}>{error} <button className={styles.button} onClick={refresh}>Retry</button></div>}
    <div className={styles.tabs} role="tablist" aria-label="Memory management">
      {TABS.map((name) => <button key={name} role="tab" aria-selected={tab === name} onClick={() => setTab(name)}>{name}</button>)}
    </div>
    {context && <div role="tabpanel">
      {tab === "Sources" && <Sources context={context} />}
      {tab === "Agent access" && <Access context={context} refresh={refresh} />}
      {tab === "Promote memory" && <Promote context={context} refresh={refresh} />}
      {tab === "Team" && <Team context={context} refresh={refresh} />}
      {tab === "Activity" && <Activity context={context} />}
    </div>}
  </div>;
}

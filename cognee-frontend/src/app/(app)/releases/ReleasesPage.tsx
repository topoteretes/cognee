"use client";

import React from "react";
import { TrackPageView } from "@/modules/analytics";
import { AsciiFrame } from "@/app/(app)/dashboard/partials/redesign/AsciiFrame";
import { FONT, LABEL, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { displayVersion, formatReleaseDate } from "@/modules/releases/releaseLog";
import { useReleases } from "@/modules/releases/useReleases";
import type { Release } from "@/modules/releases/releaseLog";

const SECTION_TITLE: React.CSSProperties = { ...LABEL, color: T.faint, letterSpacing: "0.1em", margin: "14px 0 6px" };
const LIST: React.CSSProperties = { margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.55, color: T.text };
const NOTE: React.CSSProperties = { ...FONT, fontSize: 13, color: T.muted, margin: 0 };

function Section({ title, items }: { title: string; items: string[] }): React.ReactElement | null {
  if (items.length === 0) return null;
  return (
    <>
      <div style={SECTION_TITLE}>{title}</div>
      <ul style={LIST}>
        {items.map((item) => (
          <li key={item} style={{ marginBottom: 4 }}>{item}</li>
        ))}
      </ul>
    </>
  );
}

export function ReleaseCard({ release }: { release: Release }): React.ReactElement {
  return (
    <AsciiFrame label={displayVersion(release.version) ?? release.version} meta={formatReleaseDate(release.releasedAt)} style={{ marginBottom: 16 }}>
      <div style={{ padding: "14px 16px 16px" }}>
        {release.hasNotes ? (
          <>
            {release.summary && <p style={{ ...NOTE, color: T.text, marginBottom: 4 }}>{release.summary}</p>}
            <Section title="Highlights" items={release.highlights} />
            <Section title="Fixes" items={release.fixes} />
            {release.highlights.length === 0 && release.fixes.length === 0 && (
              <p style={NOTE}>No user-visible changes in this release.</p>
            )}
          </>
        ) : (
          <p style={NOTE}>No release notes were recorded for this version.</p>
        )}
      </div>
    </AsciiFrame>
  );
}

export default function ReleasesPage(): React.ReactElement {
  const { data, isLoading, isError } = useReleases();
  const running = displayVersion(data?.runningVersion);

  let body: React.ReactNode;
  if (isLoading) {
    body = <p style={NOTE}>Loading release notes…</p>;
  } else if (isError || !data) {
    body = <p style={NOTE}>Release notes could not be loaded. Try again in a moment.</p>;
  } else if (!data.available) {
    body = <p style={NOTE}>Release notes are not available in this environment.</p>;
  } else if (data.releases.length === 0) {
    body = <p style={NOTE}>No releases recorded yet.</p>;
  } else {
    body = (
      <>
        {data.stale && (
          <p style={{ ...NOTE, color: T.amber, marginBottom: 16 }}>Showing the last copy that could be loaded; newer notes may be missing.</p>
        )}
        {data.releases.map((release) => (
          <ReleaseCard key={release.version} release={release} />
        ))}
      </>
    );
  }

  return (
    <div style={{ ...FONT, maxWidth: 760, margin: "0 auto", padding: "32px 16px 64px", color: T.text }}>
      <TrackPageView page="Release notes" />
      <div style={LABEL}>Cloud</div>
      <h1 style={{ fontSize: 24, fontWeight: 600, margin: "6px 0 4px" }}>Release notes</h1>
      <p style={{ ...NOTE, marginBottom: 24 }}>
        {running ? `This workspace runs Cloud ${running}.` : "What changed in Cloud, release by release."}
      </p>
      {body}
    </div>
  );
}

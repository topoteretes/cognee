// BusinessPage (now mounted here, see page.tsx) owns its own loading state
// (BusinessLoading) once CustomAppShell clears the pod-dependent gate, same
// as the business route it was copied from — that route has no loading.tsx
// at all. Rendering PageLoading here too stacked a second, mismatched
// "Mindmap" spinner in front of it during the route's Suspense phase. Kept
// as an empty boundary rather than deleted, since LegacyMindmapPage.tsx is
// still on disk for a possible revert.
export default function Loading() {
  return null;
}

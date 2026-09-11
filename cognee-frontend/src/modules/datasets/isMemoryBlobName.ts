// Text remembered via the memory API is stored as "text_<md5>.txt" — a
// meaningless hash filename. Detect those so we can render a proper title.
export default function isMemoryBlobName(name: string): boolean {
  return /^text_[0-9a-f]{16,}(\.txt)?$/i.test(name);
}
